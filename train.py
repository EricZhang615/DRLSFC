from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import os

import tensorflow as tf
import tensorflow_probability as tfp
import networkx as nx
import numpy as np
import tf_agents.metrics.tf_metrics

import sfcsim
from datetime import datetime

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.specs import tensor_spec
from tf_agents.networks import network, q_network, sequential
from tf_agents.agents.dqn import dqn_agent
from tf_agents.utils import common

from tf_agents.drivers import dynamic_step_driver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.policies import py_policy
from tf_agents.policies import random_py_policy
from tf_agents.policies import scripted_py_policy

from tf_agents.policies import tf_policy
from tf_agents.policies import random_tf_policy
from tf_agents.policies import epsilon_greedy_policy
from tf_agents.policies import policy_saver
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metric
from tf_agents.metrics import tf_metrics

from environment import NFVEnv

num_episodes = 150  # @param {type:"integer"}
num_itr_per_episode = 46

initial_collect_steps = 500  # @param {type:"integer"}
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 30000  # @param {type:"integer"}

batch_size = 128  # @param {type:"integer"}
shuffle = 64
learning_rate = 0.001  # @param {type:"number"}
epsilon = 0.1
target_update_tau = 0.95
target_update_period = 500
discount_gamma = 0.9

num_parallel_calls = 8
num_prefetch = batch_size

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}
log_interval = 1  # @param {type:"integer"}

checkpoint_dir = os.path.join('checkpoint/' + datetime.now().strftime("%Y%m%d-%H%M%S"), 'checkpoint')
policy_dir = os.path.join('models/' + datetime.now().strftime("%Y%m%d-%H%M%S"), 'policy')
log_dir = os.path.join('data/log', datetime.now().strftime("%Y%m%d-%H%M%S"))

train_py_env = NFVEnv(num_itr_per_episode)
eval_py_env = NFVEnv(num_itr_per_episode)
init_py_env = NFVEnv(num_itr_per_episode)

train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
init_env = tf_py_environment.TFPyEnvironment(init_py_env)

fc_layer_params = (512, 256, 128)
action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

train_summary_writer = tf.summary.create_file_writer(log_dir, flush_millis=10000)
train_summary_writer.set_as_default()


# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
    return tf.keras.layers.Dense(
        num_units,
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.RandomUniform(
            minval=-0.05, maxval=0.05, seed=None))


# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# its output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.compat.v1.train.get_or_create_global_step()

def observation_action_splitter(obs):
    return obs['observation'], obs['valid_actions']

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    # target_update_tau=target_update_tau,
    observation_and_action_constraint_splitter=observation_action_splitter,
    target_update_period=target_update_period,
    gamma=discount_gamma,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()

# replay buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

# Add an observer that adds to the replay buffer:
replay_observer = [replay_buffer.add_batch]
random_policy = random_tf_policy.RandomTFPolicy(
    init_env.time_step_spec(),
    init_env.action_spec(),
    observation_and_action_constraint_splitter=observation_action_splitter
)
initial_collect_op = dynamic_step_driver.DynamicStepDriver(
    init_env,
    random_policy,
    observers=replay_observer,
    num_steps=collect_steps_per_iteration
)

# initial collect data
time_step = init_env.reset()
step = 0
while step < initial_collect_steps or not time_step.is_last():
    step += 1
    time_step, _ = initial_collect_op.run(time_step)
# print(replay_buffer.num_frames())
dataset = replay_buffer.as_dataset(
    num_parallel_calls=num_parallel_calls,
    sample_batch_size=batch_size,
    num_steps=2
).shuffle(shuffle).prefetch(num_prefetch)
iterator = iter(dataset)

# train driver
train_policy = epsilon_greedy_policy.EpsilonGreedyPolicy(agent.policy, epsilon=epsilon)
train_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    train_policy,
    observers=replay_observer,
    num_steps=collect_steps_per_iteration
)

train_checkpoint = common.Checkpointer(
    ckpt_dir=checkpoint_dir,
    max_to_keep=1,
    agent=agent,
    policy=agent.policy,
    replay_buffer=replay_buffer,
    global_step=train_step_counter
)

train_policy_saver = policy_saver.PolicySaver(agent.policy)

train_checkpoint.initialize_or_restore()

total_step = 0
# main training loop
for episode in range(num_episodes):
    total_loss = 0
    total_reward = 0
    step = 0
    for itr in range(num_itr_per_episode * 3):

        time_step = train_env.current_time_step()

        while not time_step.is_last():
            # Collect a few steps and save to the replay buffer.
            time_step, _ = train_driver.run(time_step)
            # Sample a batch of data from the buffer and update the agent's network.
            experience, unused_info = next(iterator)
            train_loss = agent.train(experience).loss
            total_loss += train_loss
            total_reward += time_step.reward.numpy()[0]
            step += 1
            total_step += 1

        train_env.reset()
        if train_env.pyenv.get_info()['dep_fin'][0]:
            break

    # save this episode's data
    train_checkpoint.save(train_step_counter)

    if episode % log_interval == 0:
        print('Episode {}, Total step {}, episode total reward: {}, loss: {}'.format(episode, total_step, total_reward,
                                                                                     total_loss / step))
        tf.summary.scalar('episode total reward', total_reward, step=train_step_counter)
        tf.summary.scalar('episode deployed percent', train_env.pyenv.get_info()['dep_percent'][0],
                          step=train_step_counter)

train_policy_saver.save(policy_dir)


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

# compute_avg_return(eval_env, random_policy, num_eval_episodes)
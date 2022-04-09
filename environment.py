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


fail_reward = -0.5
success_reward = 1.5
delay_reward_discount = 0.5

scheduler_log = False
max_network_bw = 10.0
max_network_delay = 2.0
max_network_cpu = 10.0
max_nf_bw = 0.5*0.8*1.5  # max bw*ratio*num
max_nf_cpu = max_nf_bw*2     # max nf_bw*rec_coef
max_nf_delay = 5.0



class NFVEnv(py_environment.PyEnvironment):

    def __init__(self, num_sfc=100):
        super().__init__()
        self._dep_fin = False
        self._dep_percent = 0.0
        self._num_sfc = num_sfc
        self.network = sfcsim.cernnet2_train(num_sfc=num_sfc)
        self.scheduler = sfcsim.scheduler(log=scheduler_log)
        self.network_matrix = sfcsim.network_matrix()
        self._node_num = self.network.get_number()
        self._node_resource_attr_num = 1
        self._sfc_index = 0
        self._sfc_proc = self.network.sfcs.sfcs[self._sfc_index]     # processing sfc
        self._sfc_in_node = self._sfc_proc.get_in_node()
        self._sfc_out_node = self._sfc_proc.get_out_node()
        self._vnf_list = self._sfc_proc.get_nfs()       # list of vnfs in order
        self._vnf_index = 0
        self._vnf_proc = self._vnf_list[self._vnf_index]        # next vnf
        self._vnf_detail = self._sfc_proc.get_nfs_detail()      # next vnf attr
        self._sfc_bw = self._sfc_proc.get_bandwidths()
        self._sfc_delay = self._sfc_proc.get_delay()        # remaining delay of sfc
        self._sfc_deployed = 0

        self._node_last = None
        self._node_proc = None

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self._node_num-1, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._node_num * self._node_num + 2*self._node_num + 4, ), dtype=np.float32, minimum=0.0, name='observation'
        )

        self.network_matrix.generate(self.network)
        b = np.array([], dtype=np.float32)
        for i in range(self._node_num-1):
            b = np.append(b, (self.network_matrix.get_edge_att('remain_bandwidth')[i][i+1:]) / max_network_bw)
        d = np.array([], dtype=np.float32)
        for i in range(self._node_num-1):
            d = np.append(d, (self.network_matrix.get_edge_att('delay')[i][i+1:]) / max_network_delay)
        rsc = np.array((self.network_matrix.get_node_atts('cpu')), dtype=np.float32) / max_network_cpu
        in_node = np.zeros(self._node_num, dtype=np.float32)
        in_node[self.network_matrix.get_node_list().index(self._sfc_in_node)] = 1.0
        out_node = np.zeros(self._node_num, dtype=np.float32)
        out_node[self.network_matrix.get_node_list().index(self._sfc_out_node)] = 1.0

        self._state = np.concatenate((b, d, rsc, np.array([self._sfc_bw[self._vnf_index]/max_nf_bw], dtype=np.float32),
                                      np.array([self._vnf_detail[self._vnf_proc]['cpu']/max_nf_cpu], dtype=np.float32),
                                      np.array([self._sfc_delay/max_nf_delay], dtype=np.float32),
                                      np.array([1.0], dtype=np.float32),
                                      in_node,
                                      out_node
                                      ), dtype=np.float32)
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self, full_reset=False):
        if self._sfc_index == (self.network.sfcs.get_number()) or full_reset == True:
            # 全部sfc部署完成 清空网络开始下一组
            print('Deployed {} / {}, clearing'.format(self._sfc_deployed, self.network.sfcs.get_number()))
            self._dep_fin = True
            self._dep_percent = self._sfc_deployed / self.network.sfcs.get_number()
            # self.scheduler.show()
            self.network = sfcsim.cernnet2_train(num_sfc=self._num_sfc)
            self.scheduler = sfcsim.scheduler(log=scheduler_log)
            self.network_matrix = sfcsim.network_matrix()
            self._node_num = self.network.get_number()
            self._node_resource_attr_num = 1
            self._sfc_index = 0
            self._sfc_proc = self.network.sfcs.sfcs[self._sfc_index]  # processing sfc
            self._sfc_in_node = self._sfc_proc.get_in_node()
            self._sfc_out_node = self._sfc_proc.get_out_node()
            self._vnf_list = self._sfc_proc.get_nfs()  # list of vnfs in order
            self._vnf_index = 0
            self._vnf_proc = self._vnf_list[self._vnf_index]  # next vnf
            self._vnf_detail = self._sfc_proc.get_nfs_detail()  # next vnf attr
            self._sfc_bw = self._sfc_proc.get_bandwidths()
            self._sfc_delay = self._sfc_proc.get_delay()  # remaining delay of sfc
            self._sfc_deployed = 0

            self.network_matrix.generate(self.network)
            b = np.array([], dtype=np.float32)
            for i in range(self._node_num - 1):
                b = np.append(b, (self.network_matrix.get_edge_att('remain_bandwidth')[i][i + 1:]) / max_network_bw)
            d = np.array([], dtype=np.float32)
            for i in range(self._node_num - 1):
                d = np.append(d, (self.network_matrix.get_edge_att('delay')[i][i + 1:]) / max_network_delay)
            rsc = np.array((self.network_matrix.get_node_atts('cpu')), dtype=np.float32) / max_network_cpu
            in_node = np.zeros(self._node_num, dtype=np.float32)
            in_node[self.network_matrix.get_node_list().index(self._sfc_in_node)] = 1.0
            out_node = np.zeros(self._node_num, dtype=np.float32)
            out_node[self.network_matrix.get_node_list().index(self._sfc_out_node)] = 1.0

            self._state = np.concatenate(
                (b, d, rsc, np.array([self._sfc_bw[self._vnf_index] / max_nf_bw], dtype=np.float32),
                 np.array([self._vnf_detail[self._vnf_proc]['cpu'] / max_nf_cpu], dtype=np.float32),
                 np.array([self._sfc_delay / max_nf_delay], dtype=np.float32),
                 np.array([1.0], dtype=np.float32),
                 in_node,
                 out_node
                 ), dtype=np.float32)
            self._episode_ended = False
            return ts.restart(self._state)
        else:
            # 部署下一组sfc
            # self._sfc_index += 1
            self._sfc_proc = self.network.sfcs.sfcs[self._sfc_index]  # processing sfc
            self._sfc_in_node = self._sfc_proc.get_in_node()
            self._sfc_out_node = self._sfc_proc.get_out_node()
            self._vnf_list = self._sfc_proc.get_nfs()  # list of vnfs in order
            self._vnf_index = 0
            self._vnf_proc = self._vnf_list[self._vnf_index]  # next vnf
            self._vnf_detail = self._sfc_proc.get_nfs_detail()  # next vnf attr
            self._sfc_bw = self._sfc_proc.get_bandwidths()
            self._sfc_delay = self._sfc_proc.get_delay()  # remaining delay of sfc

            self.network_matrix.generate(self.network)
            b = np.array([], dtype=np.float32)
            for i in range(self._node_num - 1):
                b = np.append(b, (self.network_matrix.get_edge_att('remain_bandwidth')[i][i + 1:]) / max_network_bw)
            d = np.array([], dtype=np.float32)
            for i in range(self._node_num - 1):
                d = np.append(d, (self.network_matrix.get_edge_att('delay')[i][i + 1:]) / max_network_delay)
            rsc = np.array((self.network_matrix.get_node_atts('cpu')), dtype=np.float32) / max_network_cpu
            in_node = np.zeros(self._node_num, dtype=np.float32)
            in_node[self.network_matrix.get_node_list().index(self._sfc_in_node)] = 1.0
            out_node = np.zeros(self._node_num, dtype=np.float32)
            out_node[self.network_matrix.get_node_list().index(self._sfc_out_node)] = 1.0

            self._state = np.concatenate(
                (b, d, rsc, np.array([self._sfc_bw[self._vnf_index] / max_nf_bw], dtype=np.float32),
                 np.array([self._vnf_detail[self._vnf_proc]['cpu'] / max_nf_cpu], dtype=np.float32),
                 np.array([self._sfc_delay / max_nf_delay], dtype=np.float32),
                 np.array([1.0], dtype=np.float32),
                 in_node,
                 out_node
                 ), dtype=np.float32)

            self._episode_ended = False
            return ts.restart(self._state)

    def _step(self, action):

        if self._episode_ended:
            return self.reset()

        if self._dep_fin:
            self._dep_fin = False

        self.network_matrix.generate(self.network)

        if self._vnf_index == 0:
            # 是第一个vnf
            self._node_last = self.network.get_node(self._sfc_in_node)
        else:
            self._node_last = self._node_proc

        self._node_proc = self.network.get_node(self.network_matrix.get_node_list()[action])

        path = nx.shortest_path(self.network.G, source=self._node_last, target=self._node_proc, weight='delay')
        delay = nx.shortest_path_length(self.network.G, source=self._node_last, target=self._node_proc, weight='delay')
        self._sfc_delay -= (delay / max_nf_delay)
        if self._sfc_delay<0.0 or not self.scheduler.deploy_nf_scale_out(self._sfc_proc, self._node_proc, self._vnf_index + 1, self._sfc_proc.get_vnf_types()):
            # nf deploy failed
            if self._vnf_index !=0:
                self.scheduler.remove_sfc(self._sfc_proc, self.network)
            self._sfc_index += 1

            # ending this episode
            self._episode_ended = True
            return ts.termination(self._state, reward=fail_reward * (1-(self._sfc_index+1)/self.network.sfcs.get_number()))
        else:
            if not self.scheduler.deploy_link(self._sfc_proc, self._vnf_index + 1, self.network, path):
                # link deploy failed
                # remove sfc
                self.scheduler.remove_sfc(self._sfc_proc, self.network)
                self._sfc_index += 1

                # ending this episode
                self._episode_ended = True
                return ts.termination(self._state, reward=fail_reward * (1-(self._sfc_index+1)/self.network.sfcs.get_number()))
            else:
                # nf link deploy success
                if self._vnf_index < len(self._vnf_list) - 1:
                    # not last vnf to deploy
                    self._vnf_index += 1
                    self._vnf_proc = self._vnf_list[self._vnf_index]  # next vnf
                    self._vnf_detail = self._sfc_proc.get_nfs_detail()  # next vnf attr

                    self.network_matrix.generate(self.network)

                    b = np.array([], dtype=np.float32)
                    for i in range(self._node_num - 1):
                        b = np.append(b, (
                        self.network_matrix.get_edge_att('remain_bandwidth')[i][i + 1:]) / max_network_bw)
                    d = np.array([], dtype=np.float32)
                    for i in range(self._node_num - 1):
                        d = np.append(d, (self.network_matrix.get_edge_att('delay')[i][i + 1:]) / max_network_delay)
                    rsc = np.array((self.network_matrix.get_node_atts('cpu')), dtype=np.float32) / max_network_cpu

                    self._state = np.concatenate(
                        (b, d, rsc, np.array([self._sfc_bw[self._vnf_index] / max_nf_bw], dtype=np.float32),
                         np.array([self._vnf_detail[self._vnf_proc]['cpu'] / max_nf_cpu], dtype=np.float32),
                         np.array([self._sfc_delay], dtype=np.float32),
                         np.array([self._state[-(2*self._node_num+1)]-(1.0/len(self._vnf_list))], dtype=np.float32),
                         self._state[-(2*self._node_num):]
                         ), dtype=np.float32)

                    return ts.transition(self._state, reward=0.0)

                else:
                    # last vnf, deploy the last link
                    self._node_last = self._node_proc
                    self._node_proc = self.network.get_node(self._sfc_out_node)
                    path = nx.shortest_path(self.network.G, source=self._node_last, target=self._node_proc,
                                            weight='delay')
                    delay = nx.shortest_path_length(self.network.G, source=self._node_last, target=self._node_proc,
                                                    weight='delay')
                    self._sfc_delay -= (delay / max_nf_delay)
                    if self._sfc_delay<0.0 or not self.scheduler.deploy_link(self._sfc_proc, self._vnf_index+2, self.network, path):
                        # link deploy failed
                        # remove sfc
                        self.scheduler.remove_sfc(self._sfc_proc, self.network)
                        self._sfc_index += 1

                        # ending this episode
                        self._episode_ended = True
                        return ts.termination(self._state, reward=fail_reward * (1-(self._sfc_index+1)/self.network.sfcs.get_number()))
                    else:
                        # sfc deploy success

                        self.network_matrix.generate(self.network)

                        b = np.array([], dtype=np.float32)
                        for i in range(self._node_num - 1):
                            b = np.append(b, (
                                self.network_matrix.get_edge_att('remain_bandwidth')[i][i + 1:]) / max_network_bw)
                        d = np.array([], dtype=np.float32)
                        for i in range(self._node_num - 1):
                            d = np.append(d, (self.network_matrix.get_edge_att('delay')[i][i + 1:]) / max_network_delay)
                        rsc = np.array((self.network_matrix.get_node_atts('cpu')), dtype=np.float32) / max_network_cpu

                        self._state = np.concatenate(
                            (b, d, rsc, np.array([self._state[-6]], dtype=np.float32),
                             np.array([self._state[-5]], dtype=np.float32),
                             np.array([self._sfc_delay], dtype=np.float32),
                             np.array([0.0], dtype=np.float32),
                             self._state[-(2*self._node_num):]
                             ), dtype=np.float32)
                        self._sfc_index += 1
                        self._sfc_deployed += 1

                        # ending this episode
                        self._episode_ended = True
                        return ts.termination(self._state, reward=success_reward * (self._sfc_deployed/self.network.sfcs.get_number()) - delay_reward_discount*(1-self._sfc_delay))

    def get_info(self):
        return {
            'dep_fin': self._dep_fin,
            'dep_percent': self._dep_percent
        }

if __name__ == '__main__':

    # environment = NFVEnv()
    # utils.validate_py_environment(environment, episodes=5)

    num_episodes = 100  # @param {type:"integer"}
    num_itr_per_episode = 200

    initial_collect_steps = 500  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 30000  # @param {type:"integer"}

    batch_size = 64  # @param {type:"integer"}
    shuffle = 32
    learning_rate = 0.0005  # @param {type:"number"}
    epsilon = 0.1
    target_update_tau = 0.95
    target_update_period = 500
    discount_gamma = 0.9

    num_parallel_calls = 8
    num_prefetch = batch_size

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}
    log_interval = 1  # @param {type:"integer"}

    checkpoint_dir = os.path.join('checkpoint/'+datetime.now().strftime("%Y%m%d-%H%M%S"), 'checkpoint')
    policy_dir = os.path.join('models/'+datetime.now().strftime("%Y%m%d-%H%M%S"), 'policy')
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

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        # target_update_tau=target_update_tau,
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
    random_policy = random_tf_policy.RandomTFPolicy(init_env.time_step_spec(), init_env.action_spec())
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
        for itr in range(num_itr_per_episode):

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
            print('Episode {}, Total step {}, episode total reward: {}, loss: {}'.format(episode, total_step, total_reward, total_loss / step))
            tf.summary.scalar('episode total reward', total_reward, step=train_step_counter)
            tf.summary.scalar('episode deployed percent', train_env.pyenv.get_info()['dep_percent'][0], step=train_step_counter)

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



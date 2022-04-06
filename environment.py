from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import tensorflow_probability as tfp
import networkx as nx
import numpy as np
import sfcsim

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
from tf_agents.policies import actor_policy
from tf_agents.policies import q_policy
from tf_agents.policies import greedy_policy
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

discount = 1.0

fail_reward = -1
success_reward = 1

batch_size = 64


class NFVEnv(py_environment.PyEnvironment):

    def __init__(self):
        super().__init__()
        self.network = sfcsim.cernnet2()
        self.scheduler = sfcsim.scheduler()
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

        self._node_last = None
        self._node_proc = None

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=self._node_num-1, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._node_num * self._node_num + 6,), dtype=np.float32, minimum=0.0, name='observation'
        )

        self.network_matrix.generate(self.network)
        b = np.array([], dtype=np.float32)
        for i in range(self._node_num-1):
            b = np.append(b, self.network_matrix.get_edge_att('remain_bandwidth')[i][i+1:])
        d = np.array([], dtype=np.float32)
        for i in range(self._node_num-1):
            d = np.append(d, self.network_matrix.get_edge_att('delay')[i][i+1:])
        rsc = np.array(self.network_matrix.get_node_atts('cpu'), dtype=np.float32)

        self._state = np.concatenate((b, d, rsc, np.array([self._sfc_bw[self._vnf_index]], dtype=np.float32),
                                      np.array([self._vnf_detail[self._vnf_proc]['cpu']], dtype=np.float32),
                                      np.array([self._sfc_delay], dtype=np.float32),
                                      np.array([len(self._vnf_list)], dtype=np.float32),
                                      np.array([self.network_matrix.get_node_list().index(self._sfc_in_node)+1], dtype=np.float32),
                                      np.array([self.network_matrix.get_node_list().index(self._sfc_out_node)+1], dtype=np.float32)
                                      ), dtype=np.float32)
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        if self._sfc_index == (self.network.sfcs.get_number()-1):
            # 全部sfc部署完成 清空网络开始下一组
            print('finished, clearing')
            self.network = sfcsim.cernnet2()
            self.scheduler = sfcsim.scheduler()
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

            self.network_matrix.generate(self.network)
            b = np.array([], dtype=np.float32)
            for i in range(self._node_num - 1):
                b = np.append(b, self.network_matrix.get_edge_att('remain_bandwidth')[i][i + 1:])
            d = np.array([], dtype=np.float32)
            for i in range(self._node_num - 1):
                d = np.append(d, self.network_matrix.get_edge_att('delay')[i][i + 1:])
            rsc = np.array(self.network_matrix.get_node_atts('cpu'), dtype=np.float32)

            self._state = np.concatenate((b, d, rsc, np.array([self._sfc_bw[self._vnf_index]], dtype=np.float32),
                                          np.array([self._vnf_detail[self._vnf_proc]['cpu']], dtype=np.float32),
                                          np.array([self._sfc_delay], dtype=np.float32),
                                          np.array([len(self._vnf_list)], dtype=np.float32),
                                          np.array([self.network_matrix.get_node_list().index(self._sfc_in_node) + 1],
                                                   dtype=np.float32),
                                          np.array([self.network_matrix.get_node_list().index(self._sfc_out_node) + 1],
                                                   dtype=np.float32)
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
                b = np.append(b, self.network_matrix.get_edge_att('remain_bandwidth')[i][i + 1:])
            d = np.array([], dtype=np.float32)
            for i in range(self._node_num - 1):
                d = np.append(d, self.network_matrix.get_edge_att('delay')[i][i + 1:])
            rsc = np.array(self.network_matrix.get_node_atts('cpu'), dtype=np.float32)

            self._state = np.concatenate((b, d, rsc, np.array([self._sfc_bw[self._vnf_index]], dtype=np.float32),
                                          np.array([self._vnf_detail[self._vnf_proc]['cpu']], dtype=np.float32),
                                          np.array([self._sfc_delay], dtype=np.float32),
                                          np.array([len(self._vnf_list)], dtype=np.float32),
                                          np.array([self.network_matrix.get_node_list().index(self._sfc_in_node) + 1],
                                                   dtype=np.float32),
                                          np.array([self.network_matrix.get_node_list().index(self._sfc_out_node) + 1],
                                                   dtype=np.float32)
                                          ), dtype=np.float32)

            self._episode_ended = False
            return ts.restart(self._state)

    def _step(self, action):

        if self._episode_ended:
            return self.reset()

        self.network_matrix.generate(self.network)

        if self._vnf_index == 0:
            # 是第一个vnf
            self._node_last = self.network.get_node(self._sfc_in_node)
        else:
            self._node_last = self._node_proc

        self._node_proc = self.network.get_node(self.network_matrix.get_node_list()[action])

        path = nx.shortest_path(self.network.G, source=self._node_last, target=self._node_proc, weight='delay')
        delay = nx.shortest_path_length(self.network.G, source=self._node_last, target=self._node_proc, weight='delay')
        self._sfc_delay -= delay
        if self._sfc_delay<0.0 or not self.scheduler.deploy_nf_scale_out(self._sfc_proc, self._node_proc, self._vnf_index + 1, self._sfc_proc.get_vnf_types()):
            # nf deploy failed
            if self._vnf_index !=0:
                self.scheduler.remove_sfc(self._sfc_proc, self.network)
            self._sfc_index += 1

            # ending this episode
            self._episode_ended = True
            return ts.termination(self._state, reward=fail_reward)
        else:
            if not self.scheduler.deploy_link(self._sfc_proc, self._vnf_index + 1, self.network, path):
                # link deploy failed
                # remove sfc
                self.scheduler.remove_sfc(self._sfc_proc, self.network)
                self._sfc_index += 1

                # ending this episode
                self._episode_ended = True
                return ts.termination(self._state, reward=fail_reward)
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
                        b = np.append(b, self.network_matrix.get_edge_att('remain_bandwidth')[i][i + 1:])
                    d = np.array([], dtype=np.float32)
                    for i in range(self._node_num - 1):
                        d = np.append(d, self.network_matrix.get_edge_att('delay')[i][i + 1:])
                    rsc = np.array(self.network_matrix.get_node_atts('cpu'), dtype=np.float32)

                    self._state = np.concatenate(
                        (b, d, rsc, np.array([self._sfc_bw[self._vnf_index]], dtype=np.float32),
                         np.array([self._vnf_detail[self._vnf_proc]['cpu']], dtype=np.float32),
                         np.array([self._sfc_delay], dtype=np.float32),
                         np.array([self._state[-3]-1.0], dtype=np.float32),
                         np.array([self._state[-2]], dtype=np.float32),
                         np.array([self._state[-1]], dtype=np.float32)
                         ), dtype=np.float32)

                    return ts.transition(self._state, reward=0.0, discount=discount)

                else:
                    # last vnf, deploy the last link
                    self._node_last = self._node_proc
                    self._node_proc = self.network.get_node(self._sfc_out_node)
                    path = nx.shortest_path(self.network.G, source=self._node_last, target=self._node_proc,
                                            weight='delay')
                    delay = nx.shortest_path_length(self.network.G, source=self._node_last, target=self._node_proc,
                                                    weight='delay')
                    self._sfc_delay -= delay
                    if self._sfc_delay<0.0 or not self.scheduler.deploy_link(self._sfc_proc, self._vnf_index+2, self.network, path):
                        # link deploy failed
                        # remove sfc
                        self.scheduler.remove_sfc(self._sfc_proc, self.network)
                        self._sfc_index += 1

                        # ending this episode
                        self._episode_ended = True
                        return ts.termination(self._state, reward=fail_reward)
                    else:
                        # sfc deploy success

                        self.network_matrix.generate(self.network)

                        b = np.array([], dtype=np.float32)
                        for i in range(self._node_num - 1):
                            b = np.append(b, self.network_matrix.get_edge_att('remain_bandwidth')[i][i + 1:])
                        d = np.array([], dtype=np.float32)
                        for i in range(self._node_num - 1):
                            d = np.append(d, self.network_matrix.get_edge_att('delay')[i][i + 1:])
                        rsc = np.array(self.network_matrix.get_node_atts('cpu'), dtype=np.float32)

                        self._state = np.concatenate(
                            (b, d, rsc, np.array([self._state[-6]], dtype=np.float32),
                             np.array([self._state[-5]], dtype=np.float32),
                             np.array([self._sfc_delay], dtype=np.float32),
                             np.array([self._state[-3] - 1.0], dtype=np.float32),
                             np.array([self._state[-2]], dtype=np.float32),
                             np.array([self._state[-1]], dtype=np.float32)
                             ), dtype=np.float32)
                        self._sfc_index += 1

                        # ending this episode
                        self._episode_ended = True
                        return ts.termination(self._state, reward=success_reward)


if __name__ == '__main__':
    num_iterations = 20000  # @param {type:"integer"}

    initial_collect_steps = 100  # @param {type:"integer"}
    collect_steps_per_iteration = 1  # @param {type:"integer"}
    replay_buffer_max_length = 100000  # @param {type:"integer"}

    batch_size = 16  # @param {type:"integer"}
    learning_rate = 1e-3  # @param {type:"number"}
    log_interval = 200  # @param {type:"integer"}

    num_eval_episodes = 10  # @param {type:"integer"}
    eval_interval = 1000  # @param {type:"integer"}

    train_py_env = NFVEnv()
    eval_py_env = NFVEnv()

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    fc_layer_params = (100, 50)
    action_tensor_spec = tensor_spec.from_spec(train_env.action_spec())
    num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1


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

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    eval_policy = agent.policy
    collect_policy = agent.collect_policy
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())

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

    # See also the metrics module for standard implementations of different metrics.
    # https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

    compute_avg_return(eval_env, random_policy, num_eval_episodes)

    # environment = NFVEnv()
    # utils.validate_py_environment(environment, episodes=5)


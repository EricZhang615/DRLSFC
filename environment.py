from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import ctypes
import os
import multiprocessing

import tensorflow as tf
import tensorflow_probability as tfp
import networkx as nx
import numpy as np
import tf_agents.metrics.tf_metrics

import sfcsim
from datetime import datetime
from copy import deepcopy

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
        self._dep_attempts = 0
        # self.network = sfcsim.cernnet2_train(num_sfc=num_sfc)
        self.network = sfcsim.cernnet2()
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
        self._observation_spec = {
            'observation': array_spec.BoundedArraySpec(
                shape=(self._node_num * self._node_num + 2 * self._node_num + 4,),
                dtype=np.float32,
                minimum=0.0,
                name='observation'
            ),
            'valid_actions': array_spec.ArraySpec(
                shape=(self._node_num,),
                dtype=np.bool_,
                name='valid_actions'
            )
        }

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

        obs = {
            'observation': np.concatenate(
            (b, d, rsc, np.array([self._sfc_bw[self._vnf_index] / max_nf_bw], dtype=np.float32),
             np.array([self._vnf_detail[self._vnf_proc]['cpu'] / max_nf_cpu], dtype=np.float32),
             np.array([self._sfc_delay / max_nf_delay], dtype=np.float32),
             np.array([1.0], dtype=np.float32),
             in_node,
             out_node
             ), dtype=np.float32),
            'valid_actions': self.get_valid_actions()
        }
        self._state = obs
        self._episode_ended = False

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self, full_reset=False):
        if self._sfc_index == (self.network.sfcs.get_number()) or self._dep_attempts == self._num_sfc*3 or full_reset == True:
            # 全部sfc部署完成 清空网络开始下一组
            print('Deployed {} / {}, {} attempts'.format(self._sfc_deployed, self._num_sfc, self._dep_attempts))
            self._dep_fin = True
            self._dep_percent = self._sfc_deployed / self._num_sfc
            self._dep_attempts = 0
            # self.scheduler.show()
            # self.network = sfcsim.cernnet2_train(num_sfc=self._num_sfc)
            self.network = sfcsim.cernnet2()
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

            obs = {
                'observation': np.concatenate(
                (b, d, rsc, np.array([self._sfc_bw[self._vnf_index] / max_nf_bw], dtype=np.float32),
                 np.array([self._vnf_detail[self._vnf_proc]['cpu'] / max_nf_cpu], dtype=np.float32),
                 np.array([self._sfc_delay / max_nf_delay], dtype=np.float32),
                 np.array([1.0], dtype=np.float32),
                 in_node,
                 out_node
                 ), dtype=np.float32),
                'valid_actions': self.get_valid_actions()
            }
            self._state = obs
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

            obs = {
                'observation': np.concatenate(
                (b, d, rsc, np.array([self._sfc_bw[self._vnf_index] / max_nf_bw], dtype=np.float32),
                 np.array([self._vnf_detail[self._vnf_proc]['cpu'] / max_nf_cpu], dtype=np.float32),
                 np.array([self._sfc_delay / max_nf_delay], dtype=np.float32),
                 np.array([1.0], dtype=np.float32),
                 in_node,
                 out_node
                 ), dtype=np.float32),
                'valid_actions': self.get_valid_actions()
            }
            self._state = obs
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
            self.network.sfcs.add_sfc(self.network.sfcs.pop_sfc(self._sfc_proc.get_id()))
            self._dep_attempts += 1

            # ending this episode
            self._episode_ended = True
            return ts.termination(self._state, reward=fail_reward * (1-(self._sfc_index+1)/self._num_sfc))
        else:
            if not self.scheduler.deploy_link(self._sfc_proc, self._vnf_index + 1, self.network, path):
                # link deploy failed
                # remove sfc
                self.scheduler.remove_sfc(self._sfc_proc, self.network)
                self.network.sfcs.add_sfc(self.network.sfcs.pop_sfc(self._sfc_proc.get_id()))
                self._dep_attempts += 1

                # ending this episode
                self._episode_ended = True
                return ts.termination(self._state, reward=fail_reward * (1-(self._sfc_index+1)/self._num_sfc))
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

                    obs = {
                        'observation': np.concatenate(
                        (b, d, rsc, np.array([self._sfc_bw[self._vnf_index] / max_nf_bw], dtype=np.float32),
                         np.array([self._vnf_detail[self._vnf_proc]['cpu'] / max_nf_cpu], dtype=np.float32),
                         np.array([self._sfc_delay], dtype=np.float32),
                         np.array([self._state['observation'][-(2 * self._node_num + 1)] - (1.0 / len(self._vnf_list))],
                                  dtype=np.float32),
                         self._state['observation'][-(2 * self._node_num):]
                         ), dtype=np.float32),
                        'valid_actions': self.get_valid_actions()
                    }
                    self._state = obs
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
                        self.network.sfcs.add_sfc(self.network.sfcs.pop_sfc(self._sfc_proc.get_id()))
                        self._dep_attempts += 1

                        # ending this episode
                        self._episode_ended = True
                        return ts.termination(self._state, reward=fail_reward * (1-(self._sfc_index+1)/self._num_sfc))
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

                        obs = {
                            'observation': np.concatenate(
                            (b, d, rsc, np.array([self._state['observation'][-6]], dtype=np.float32),
                             np.array([self._state['observation'][-5]], dtype=np.float32),
                             np.array([self._sfc_delay], dtype=np.float32),
                             np.array([0.0], dtype=np.float32),
                             self._state['observation'][-(2 * self._node_num):]
                             ), dtype=np.float32),
                            'valid_actions': self._state['valid_actions']
                        }
                        self._state = obs
                        self._sfc_index += 1
                        self._sfc_deployed += 1
                        self._dep_attempts += 1

                        # ending this episode
                        self._episode_ended = True
                        return ts.termination(self._state, reward=success_reward * (self._sfc_deployed/self._num_sfc) - delay_reward_discount*(1-self._sfc_delay))

    def get_info(self):
        return {
            'dep_fin': self._dep_fin,
            'dep_percent': self._dep_percent
        }

    def get_valid_actions(self):

        va = [False for _ in range(self._node_num)]
        m = multiprocessing.Manager()
        va = m.Array('B', va)
        p = multiprocessing.Pool(8)
        for action in range(self._node_num):
            p.apply_async(self.verify_action, args=(action, va))

        p.close()
        p.join()

        valid_actions = np.array(va, dtype=np.bool_)
        t = np.full((self._node_num,), False, dtype=np.bool_)
        if (valid_actions == t).all():
            valid_actions = np.full((self._node_num,), True, dtype=np.bool_)

        p.terminate()

        return valid_actions


    def verify_action(self, action, valid_actions=()):

        net = deepcopy(self.network)
        sch = deepcopy(self.scheduler)
        net_matrix = sfcsim.network_matrix()
        sfc_proc = net.sfcs.get_sfc(self._sfc_proc.get_id())
        net_matrix.generate(net)

        if self._vnf_index == 0:
            # 是第一个vnf
            node_last = net.get_node(self._sfc_in_node)
        else:
            node_last = net.get_node(self._node_proc.get_id())

        node_proc = net.get_node(net_matrix.get_node_list()[action])

        path = nx.shortest_path(net.G, source=node_last, target=node_proc, weight='delay')
        delay = nx.shortest_path_length(net.G, source=node_last, target=node_proc, weight='delay')
        sfc_delay = self._sfc_delay - (delay / max_nf_delay)
        if sfc_delay < 0.0 or not sch.deploy_nf_scale_out(sfc_proc, node_proc, self._vnf_index + 1,
                                                          sfc_proc.get_vnf_types()):
            # nf deploy failed
            # ending this attempt
            valid_actions[action] = False

        else:
            if not sch.deploy_link(sfc_proc, self._vnf_index + 1, net, path):
                # link deploy failed
                sch.remove_nf(sfc_proc, self._vnf_index + 1)
                # ending this episode
                valid_actions[action] = False

            else:
                # nf link deploy success
                if self._vnf_index < len(self._vnf_list) - 1:
                    # not last vnf to deploy
                    # remove nf and links
                    sch.remove_link(sfc_proc, self._vnf_index + 1, net)
                    sch.remove_nf(sfc_proc, self._vnf_index + 1)
                    # ending this attempt
                    valid_actions[action] = True

                else:
                    # last vnf, deploy the last link
                    node_last = node_proc
                    node_proc = net.get_node(self._sfc_out_node)
                    path = nx.shortest_path(net.G, source=node_last, target=node_proc, weight='delay')
                    delay = nx.shortest_path_length(net.G, source=node_last, target=node_proc, weight='delay')
                    sfc_delay -= (delay / max_nf_delay)
                    if sfc_delay < 0.0 or not sch.deploy_link(sfc_proc, self._vnf_index + 2, net, path):
                        # link deploy failed
                        sch.remove_link(sfc_proc, self._vnf_index + 1, net)
                        sch.remove_nf(sfc_proc, self._vnf_index + 1)
                        # ending this episode
                        valid_actions[action] = False

                    else:
                        # sfc deploy success
                        # remove nf and links
                        sch.remove_link(sfc_proc, self._vnf_index + 2, net)
                        sch.remove_link(sfc_proc, self._vnf_index + 1, net)
                        sch.remove_nf(sfc_proc, self._vnf_index + 1)
                        # ending this episode
                        valid_actions[action] = True

        del net, sch
        return 0

if __name__ == '__main__':

    environment = NFVEnv()
    utils.validate_py_environment(environment, episodes=5)





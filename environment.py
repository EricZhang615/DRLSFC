from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import sfcsim

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts




class NFVEnv(py_environment.PyEnvironment):

    def __init__(self):
        self.network = sfcsim.cernnet2()
        self.scheduler = sfcsim.scheduler()
        self.network_matrix = sfcsim.network_matrix()
        self._node_num = self.network.get_number()
        self._node_resource_attr_num = 1
        self._sfc_index = 0
        self._sfc_proc = self.network.sfcs[self._sfc_index]     # processing sfc
        self._sfc_in_node = self._sfc_proc.get_in_node()
        self._sfc_out_node = self._sfc_proc.get_out_node()
        self._vnf_list = self._sfc_proc.get_nfs()       # list of vnfs in order
        self._vnf_index = 0
        self._vnf_proc = self._vnf_list[self._vnf_index]        # next vnf
        self._vnf_detail = self._sfc_proc.get_nfs_detail()      # next vnf attr
        self._sfc_bw = self._sfc_proc.get_bandwidths()
        self._sfc_delay = self._sfc_proc.get_delay()        # remaining delay of sfc

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=1, maximum=self._node_num, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(8+self._node_resource_attr_num, self._node_num, self._node_num), dtype=np.float32,
            minimum=0.0, name='observation'
        )

        self.network_matrix.generate(self.network)
        in_node_s = np.zeros([self._node_num, self._node_num])
        in_node_s[..., self.network_matrix.get_node_list().index(self._sfc_in_node):(1 + self.network_matrix.get_node_list().index(self._sfc_in_node))] = 1
        out_node_s = np.zeros([self._node_num, self._node_num])
        out_node_s[..., self.network_matrix.get_node_list().index(self._sfc_out_node):(1 + self.network_matrix.get_node_list().index(self._sfc_out_node))] = 1

        self._state = np.array([self.network_matrix.get_edge_att('remain_bandwidth'),
                                self.network_matrix.get_edge_att('delay'),
                                np.array([self.network_matrix.get_node_atts('cpu')])*(np.array([np.linspace(1,1,self._node_num)]).T),
                                self._vnf_detail[self._vnf_proc]['cpu']*np.ones([self._node_num, self._node_num]),
                                self._sfc_bw[self._vnf_index]*np.ones([self._node_num, self._node_num]),
                                self._sfc_delay*np.ones([self._node_num, self._node_num]),
                                len(self._vnf_list)*np.ones([self._node_num, self._node_num]),
                                in_node_s,
                                out_node_s
                                ], dtype=np.float32)
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
            self._sfc_proc = self.network.sfcs[self._sfc_index]  # processing sfc
            self._sfc_in_node = self._sfc_proc.get_in_node()
            self._sfc_out_node = self._sfc_proc.get_out_node()
            self._vnf_list = self._sfc_proc.get_nfs()  # list of vnfs in order
            self._vnf_index = 0
            self._vnf_proc = self._vnf_list[self._vnf_index]  # next vnf
            self._vnf_detail = self._sfc_proc.get_nfs_detail()  # next vnf attr
            self._sfc_bw = self._sfc_proc.get_bandwidths()
            self._sfc_delay = self._sfc_proc.get_delay()  # remaining delay of sfc

            self.network_matrix.generate(self.network)
            in_node_s = np.zeros([self._node_num, self._node_num])
            in_node_s[..., self.network_matrix.get_node_list().index(self._sfc_in_node):(
                        1 + self.network_matrix.get_node_list().index(self._sfc_in_node))] = 1
            out_node_s = np.zeros([self._node_num, self._node_num])
            out_node_s[..., self.network_matrix.get_node_list().index(self._sfc_out_node):(
                        1 + self.network_matrix.get_node_list().index(self._sfc_out_node))] = 1

            self._state = np.array([self.network_matrix.get_edge_att('remain_bandwidth'),
                                    self.network_matrix.get_edge_att('delay'),
                                    np.array([self.network_matrix.get_node_atts('cpu')]) * (
                                        np.array([np.linspace(1, 1, self._node_num)]).T),
                                    self._vnf_detail[self._vnf_proc]['cpu'] * np.ones([self._node_num, self._node_num]),
                                    self._sfc_bw[self._vnf_index] * np.ones([self._node_num, self._node_num]),
                                    self._sfc_delay * np.ones([self._node_num, self._node_num]),
                                    len(self._vnf_list) * np.ones([self._node_num, self._node_num]),
                                    in_node_s,
                                    out_node_s
                                    ], dtype=np.float32)
            self._episode_ended = False
            return ts.restart(self._state)
        else:
            self._sfc_index += 1
            self._sfc_proc = self.network.sfcs[self._sfc_index]  # processing sfc
            self._sfc_in_node = self._sfc_proc.get_in_node()
            self._sfc_out_node = self._sfc_proc.get_out_node()
            self._vnf_list = self._sfc_proc.get_nfs()  # list of vnfs in order
            self._vnf_index = 0
            self._vnf_proc = self._vnf_list[self._vnf_index]  # next vnf
            self._vnf_detail = self._sfc_proc.get_nfs_detail()  # next vnf attr
            self._sfc_bw = self._sfc_proc.get_bandwidths()
            self._sfc_delay = self._sfc_proc.get_delay()  # remaining delay of sfc

            self.network_matrix.generate(self.network)
            in_node_s = np.zeros([self._node_num, self._node_num])
            in_node_s[..., self.network_matrix.get_node_list().index(self._sfc_in_node):(
                    1 + self.network_matrix.get_node_list().index(self._sfc_in_node))] = 1
            out_node_s = np.zeros([self._node_num, self._node_num])
            out_node_s[..., self.network_matrix.get_node_list().index(self._sfc_out_node):(
                    1 + self.network_matrix.get_node_list().index(self._sfc_out_node))] = 1

            self._state = np.array([self.network_matrix.get_edge_att('remain_bandwidth'),
                                    self.network_matrix.get_edge_att('delay'),
                                    np.array([self.network_matrix.get_node_atts('cpu')]) * (
                                        np.array([np.linspace(1, 1, self._node_num)]).T),
                                    self._vnf_detail[self._vnf_proc]['cpu'] * np.ones([self._node_num, self._node_num]),
                                    self._sfc_bw[self._vnf_index] * np.ones([self._node_num, self._node_num]),
                                    self._sfc_delay * np.ones([self._node_num, self._node_num]),
                                    len(self._vnf_list) * np.ones([self._node_num, self._node_num]),
                                    in_node_s,
                                    out_node_s
                                    ], dtype=np.float32)
            self._episode_ended = False
            return ts.restart(self._state)

    def _step(self, action):

        if self._episode_ended:
            return self.reset()

        self.network_matrix.generate(self.network)
        node=self.network_matrix.get_node_list()[action-1]

        if self._vnf_index == 0:
            # 是第一个vnf
            if self.scheduler.deploy_nf(self._sfc_proc,node,self._vnf_index+1):
                self.scheduler.deploy_link(self._sfc_proc,self._vnf_index+1,self.network,)





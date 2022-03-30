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

    def __init__(self, network_struct=sfcsim.cernnet2()):
        self.network = network_struct
        self.scheduler = sfcsim.scheduler()
        self._node_num = self.network.get_number()
        self._node_resource_attr_num = 1

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2*self._node_num-1, name='action'
        )
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(8+self._node_resource_attr_num, self._node_num, self._node_num), dtype=np.float32,
            minimum=0.0, name='observation'
        )
        self._state =

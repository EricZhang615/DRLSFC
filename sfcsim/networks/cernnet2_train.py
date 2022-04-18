from sfcsim.classes.network import *
from sfcsim.classes.sfc import *
from sfcsim.layout.cernnet2_layout import *

import random
from random import choice
class cernnet2_train(network):
    '''
    研究组实验室的开放挑战，挑战目标：在底层网络上部署文件中的sfc，算法执行时间短且总部署流量大者取胜
    属性值(cernnet继承network类，能使用network所有方法)：底层网络，网络拓扑为cernnet结构，详情见http://www.cernet20.edu.cn/introduction.shtml
        nodes               节点资源分布符合U(10~30) 
        G                   链路延迟符合U(0.5,1.5) (ms)
        vnf_types           vnf_types类实例，所有类型vnf集合，一共八种
        sfcs                sfcs类实例，所需部署目标服务功能链 
                                                       mMTC  30条 延迟分布U(5,10) 流量需求(0.1~0.5G) 长度 3~5nf
                                                       uRLLC 10条 延迟分布U(2,4) 流量需求(1~2G) 长度 1~2nf   
                                                       eMBB  6条 延迟分布U(5,10) 流量需求(3~4G) 长度 3~5nf    
    下列属性方法能够打印出底层数据结构：
        cernnet.vnf_types.show()
        cernnet.sfc.show()
        cernnet.show()
        cernnet.draw()
    '''
    def __init__(self, num_sfc=100):
        self.node1=node(uuid='node1',atts={'cpu':10,'access':False})
        self.node2=node(uuid='node2',atts={'cpu':10,'access':False})
        self.node3=node(uuid='node3',atts={'cpu':10,'access':False})
        self.node4=node(uuid='node4',atts={'cpu':10,'access':False})
        self.node5=node(uuid='node5',atts={'cpu':10,'access':False})
        self.node6=node(uuid='node6',atts={'cpu':10,'access':False})
        self.node7=node(uuid='node7',atts={'cpu':10,'access':False})
        self.node8=node(uuid='node8',atts={'cpu':10,'access':False})
        self.node9=node(uuid='node9',atts={'cpu':10,'access':False})
        self.node10=node(uuid='node10',atts={'cpu':10,'access':False})
        self.node11=node(uuid='node11',atts={'cpu':10,'access':False})
        self.node12=node(uuid='node12',atts={'cpu':10,'access':False})
        self.node13=node(uuid='node13',atts={'cpu':10,'access':False})
        self.node14=node(uuid='node14',atts={'cpu':10,'access':False})
        self.node15=node(uuid='node15',atts={'cpu':10,'access':False})
        self.node16=node(uuid='node16',atts={'cpu':10,'access':False})
        self.node17=node(uuid='node17',atts={'cpu':10,'access':False})
        self.node18=node(uuid='node18',atts={'cpu':10,'access':False})
        self.node19=node(uuid='node19',atts={'cpu':10,'access':False})
        self.node20=node(uuid='node20',atts={'cpu':10,'access':False})
        self.node21=node(uuid='node21',atts={'cpu':10,'access':False})
        server_nodes=[self.node1,self.node2,self.node3,self.node4,self.node5,self.node6,self.node7,self.node8,self.node9,self.node10,\
                   self.node11,self.node12,self.node13,self.node14,self.node15,self.node16,self.node17,self.node18,self.node19,self.node20,self.node21]
        access_nodes=[]
        network.__init__(self,server_nodes+access_nodes)
        self.generate_edges()
        self.generate_nodes_atts()
        self.generate_edges_atts()
        self.vnf_types=vnf_types(vnf_types=[(vnf_type(name='type1',atts={'cpu':0},ratio=0.8,resource_coefficient={'cpu':1}))\
                        ,vnf_type(name='type2',atts={'cpu':0},ratio=0.8,resource_coefficient={'cpu':1})\
                        ,vnf_type(name='type3',atts={'cpu':0},ratio=1.2,resource_coefficient={'cpu':1.8})\
                        ,vnf_type(name='type4',atts={'cpu':0},ratio=1.5,resource_coefficient={'cpu':1.5})\
                        ,vnf_type(name='type5',atts={'cpu':0},ratio=1,resource_coefficient={'cpu':1.4})\
                        ,vnf_type(name='type6',atts={'cpu':0},ratio=1,resource_coefficient={'cpu':1.2})\
                        ,vnf_type(name='type7',atts={'cpu':0},ratio=0.8,resource_coefficient={'cpu':1.2})\
                        ,vnf_type(name='type8',atts={'cpu':0},ratio=1,resource_coefficient={'cpu':2})])
        vnf_list = ['type2','type4','type6','type8']
        in_node_list = ['node20', 'node7', 'node1', 'node4', 'node20', 'node21', 'node20', 'node19', 'node12', 'node20',
                        'node3', 'node8', 'node18', 'node14', 'node5', 'node5', 'node16', 'node3', 'node3', 'node10',
                        'node15', 'node17', 'node6', 'node6', 'node21', 'node20', 'node17', 'node13', 'node9', 'node12',
                        'node8', 'node1', 'node13', 'node20', 'node13', 'node15', 'node8', 'node9', 'node2', 'node16',
                        'node20', 'node21', 'node20', 'node5', 'node15', 'node8', 'node14', 'node7', 'node3', 'node6',
                        'node11', 'node19', 'node19', 'node2', 'node6', 'node9', 'node6', 'node9', 'node13', 'node2',
                        'node3', 'node8', 'node18', 'node14', 'node5', 'node5', 'node16', 'node3', 'node13', 'node10',
                        'node11', 'node17', 'node19', 'node2', 'node6', 'node9', 'node6', 'node9', 'node13', 'node2',
                        'node12', 'node4', 'node3', 'node1', 'node2', 'node1', 'node13', 'node11', 'node15', 'node15',
                        'node15', 'node12', 'node13', 'node5', 'node7', 'node13', 'node5', 'node13', 'node15', 'node20',
                        'node8', 'node9', 'node9', 'node19', 'node21', 'node15', 'node6', 'node11', 'node19', 'node17',
                        'node11', 'node17', 'node19', 'node2', 'node6', 'node9', 'node6', 'node9', 'node13', 'node2',
                        'node20', 'node7', 'node1', 'node4', 'node20', 'node21', 'node20', 'node19', 'node12', 'node20',
                        'node15', 'node17', 'node6', 'node6', 'node21', 'node20', 'node17', 'node13', 'node9', 'node12',
                        'node20', 'node21', 'node20', 'node5', 'node15', 'node8', 'node14', 'node7', 'node3', 'node6',
                        'node13', 'node1', 'node20', 'node18', 'node11', 'node19', 'node17', 'node21', 'node9',
                        'node20', 'node21', 'node20', 'node5', 'node15', 'node8', 'node14', 'node7', 'node3', 'node6',
                        'node12', 'node4', 'node3', 'node1', 'node2', 'node1', 'node13', 'node11', 'node15', 'node15',
                        'node8', 'node9', 'node9', 'node19', 'node21', 'node15', 'node6', 'node11', 'node19', 'node17',
                        'node3', 'node8', 'node18', 'node14', 'node5', 'node5', 'node16', 'node3', 'node3', 'node10',
                        'node16']
        out_node_list = ['node16', 'node5', 'node5', 'node10', 'node18', 'node8', 'node18', 'node5', 'node16', 'node5',
                         'node21', 'node9', 'node14', 'node19', 'node20', 'node8', 'node9', 'node13', 'node10', 'node6',
                         'node1', 'node13', 'node5', 'node1', 'node1', 'node1', 'node3', 'node15', 'node3', 'node1',
                         'node15', 'node3', 'node6', 'node21', 'node14', 'node19', 'node9', 'node17', 'node14', 'node3',
                         'node12', 'node8', 'node5', 'node10', 'node10', 'node7', 'node19', 'node3', 'node21', 'node7',
                         'node14', 'node17', 'node17', 'node10', 'node18', 'node21', 'node12', 'node21', 'node18',
                         'node15', 'node13', 'node6', 'node21', 'node4', 'node19', 'node9', 'node17', 'node14', 'node3',
                         'node16', 'node17', 'node5', 'node18', 'node15', 'node11', 'node19', 'node20', 'node12',
                         'node8', 'node5', 'node10', 'node10', 'node7', 'node19', 'node3', 'node21', 'node7', 'node1',
                         'node21', 'node9', 'node14', 'node19', 'node20', 'node8', 'node9', 'node9', 'node10', 'node6',
                         'node16', 'node5', 'node5', 'node10', 'node18', 'node8', 'node18', 'node5', 'node16', 'node5',
                         'node2', 'node15', 'node13', 'node5', 'node1', 'node1', 'node1', 'node3', 'node15', 'node3',
                         'node1', 'node11', 'node21', 'node15', 'node10', 'node13', 'node9', 'node2', 'node2', 'node6',
                         'node17', 'node4', 'node11', 'node8', 'node13', 'node9', 'node7', 'node7', 'node4', 'node1',
                         'node20', 'node11', 'node21', 'node15', 'node10', 'node13', 'node9', 'node2', 'node2', 'node6',
                         'node17', 'node3', 'node19', 'node9', 'node4', 'node6', 'node3', 'node15', 'node6', 'node15',
                         'node2', 'node15', 'node13', 'node5', 'node1', 'node1', 'node1', 'node3', 'node15', 'node15',
                         'node1', 'node15', 'node3', 'node6', 'node21', 'node14', 'node19', 'node9', 'node17', 'node14',
                         'node3', 'node3', 'node19', 'node19', 'node4', 'node6', 'node3', 'node15', 'node6', 'node15',
                         'node2', 'node15', 'node13', 'node5', 'node1', 'node1', 'node1', 'node3', 'node15', 'node15',
                         'node1', 'node17', 'node8']
        sfc_list = []
        for i in range(1, num_sfc+1):
            # length = random.randint(3,5)
            s = sfc('sfc'+str(i),in_node_list[i-1],out_node_list[i-1],vnf_list,
                    0.5,5.0,0,0,self.vnf_types)
            sfc_list.append(s)
        self.sfcs=sfcs(sfc_list)
        self.figure=''
    def generate_edges(self):
        self.add_edges([[self.node1,self.node2,{'bandwidth':10}],[self.node2,self.node3,{'bandwidth':10}],\
                        [self.node3,self.node4,{'bandwidth':10}],[self.node3,self.node5,{'bandwidth':10}],\
                        [self.node5,self.node6,{'bandwidth':10}],[self.node5,self.node7,{'bandwidth':10}],\
                        [self.node5,self.node9,{'bandwidth':10}],[self.node5,self.node16,{'bandwidth':10}],\
                        [self.node6,self.node8,{'bandwidth':10}],[self.node7,self.node9,{'bandwidth':10}],\
                        [self.node8,self.node12,{'bandwidth':10}],[self.node9,self.node10,{'bandwidth':10}],\
                        [self.node10,self.node11,{'bandwidth':10}],[self.node12,self.node13,{'bandwidth':10}],\
                        [self.node12,self.node14,{'bandwidth':10}],[self.node13,self.node15,{'bandwidth':10}],\
                        [self.node14,self.node16,{'bandwidth':10}],[self.node15,self.node20,{'bandwidth':10}],\
                        [self.node16,self.node17,{'bandwidth':10}],[self.node16,self.node19,{'bandwidth':10}],\
                        [self.node16,self.node21,{'bandwidth':10}],[self.node17,self.node18,{'bandwidth':10}],[self.node20,self.node21,{'bandwidth':10}]])
    def generate_nodes_atts(self,atts=[30, 29, 28, 27, 27, 27, 26, 22, 22, 20, 19, 17, 16, 16, 14, 14, 13, 13, 12, 11, 10]):
        nodes=[5,16,21,3,12,13,10,1,2,4,6,7,8,9,11,14,15,17,18,19,20]
        if len(atts)==len(nodes):
            i=0
            for node in nodes:
                self.set_atts('node'+str(node),{'cpu':atts[i]})
                i+=1
    def generate_edges_atts(self,atts=[0.77, 0.59, 1.47, 0.95, 0.59, 0.69, 1.56, 1.1, 0.52, 1.03, 0.95, 1.08, 0.83, 1.21, 1.33, 0.92, 0.75, 1.34, 1.22, 1.29, 0.56, 0.64, 1.3]):
        i=0
        for edge in self.G.edges:
            self.set_edge_atts(edge[0],edge[1],{'delay':atts[i]})
            i+=1
    def draw(self,figsize=[36,20],node_size=10000,node_fone_size=8,link_fone_size=9,node_shape='H',path=''):
        network.draw(self,figsize=figsize,pos=cernnet2_layout(self.G),node_size=node_size,node_fone_size=node_fone_size,link_fone_size=link_fone_size,node_shape=node_shape)
    def draw_dynamic(self,figsize=[36,20],path='',node_size=10000,node_fone_size=8,link_fone_size=9,node_shape='H'):
        network.draw_dynamic(self,figsize=figsize,pos=cernnet2_layout(self.G),node_size=node_size,node_fone_size=node_fone_size,link_fone_size=link_fone_size,node_shape=node_shape)

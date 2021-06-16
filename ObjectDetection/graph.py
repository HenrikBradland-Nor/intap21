from config import *
from dgl.nn.pytorch.factory import KNNGraph


class graphMaster:
    def __init__(self, cfg):
        self.cfg = cfg
        self._graph = None
        self._kg = KNNGraph(self.cfg.GRAPH_K)

    def newGraphFromPointCloud(self, pc):
        # Froms graph based on cartesian coordinates
        self._graph = self._kg(pc[:, :3])

    def newGrapgFromRelationVector(self, edg_b, edg_n):
        self._graph = dgl.graph((edg_b, edg_n))

    def showGraph(self):
        nx_g = self._graph.to_networkx().to_undirected()
        pos = nx.kamada_kawai_layout(nx_g)
        nx.draw(nx_g, pos, with_labels=True, node_color=[[.7, .7, .7]])
        plt.show()

    def printGraph(self):
        print(self._graph)

    def getGraph(self):
        return self._graph
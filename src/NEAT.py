"""
Author: gabriel jiglau
StartingDate: 2024-IV-03
Description: Actual implementation of the NEAT(neuro-evolution of augmenting topologies) algorithm for Geometry Dash
"""

from BuildingBlocks import Gene, Node, Connection, NodesNotConnectedException

import random

class NEAT:
    def __int__(self, genes=None):
        if genes is None:
            self.genes = []
        else:
            self.genes = genes
"""
Author: gabriel jiglau
StartingDate: 2024-III-26
Description: All the concepts I will be using, but fragmented into smaller parts
"""
import random

# global variable used for keeping track of the gene ancestry
innovation_number = 1

class NodesNotConnectedException(Exception):
    def __init__(self, first_node, second_node):
        self.message = f"Nodes {first_node} and {second_node} are not connected."

    def __str__(self):
        return self.message

class Connection:
    def __init__(self, in_node, out_node, is_enabled):
        self.in_node = in_node
        self.out_node = out_node
        self.is_enabled = is_enabled

    def __str__(self):
        return f"Connection from Node {self.in_node} to Node {self.out_node} is enabled : {self.is_enabled}"


class Node:
    def __init__(self, weight):
        global innovation_number

        # the weight should be set randomly at first
        self.weight = weight

        # innovation_number gets incremented after each new node
        self.innovation_number = innovation_number
        innovation_number += 1
        self.connections = []

    def __str__(self):
        return f"Node(weight={self.weight}, innovation_number={self.innovation_number})"


class Gene:
    def __init__(self, nodes=None, connections=None):
        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes
        if connections is None:
            self.connections = []
        else:
            self.connections = connections

    def are_nodes_connected(self, first_node: 'Node', second_node: 'Node') -> bool:
        # connections are one directional
        for connection in self.connections:
            if connection.in_node == first_node and connection.out_node == second_node:
                return True
        return False

    def is_connection_enabled(self, source_node: 'Node', target_node: 'Node') -> bool:
        for connection in self.connections:
            if connection.in_node == source_node and connection.out_node == target_node:
                return connection.is_enabled
        return False

    # the relationship between the nodes is one-directional, to ensure that there won't be any cycles introduced
    def add_node_between_genes(self, first_node: 'Node', second_node: 'Node', node_to_add: 'Node'):

        if not self.are_nodes_connected(first_node, second_node):
            raise NodesNotConnectedException(first_node, second_node)

        # create new connections
        is_enbled = True
        first_connection = Connection(first_node, node_to_add, is_enbled)
        second_connection = Connection(node_to_add, second_node, is_enbled)

        self.connections.append(first_connection)
        self.connections.append(second_connection)

        self.nodes.append(node_to_add)

        # Update the existing connections
        for connection in self.connections:
            if connection.out_node == second_node:
                connection.out_node = node_to_add
            if connection.in_node == second_node:
                connection.in_node = node_to_add

    def add_connection(self, first_node: 'Node', second_node: 'Node'):
        # connections are one directional
        is_enabled = True
        connection = Connection(first_node, second_node, is_enabled)
        self.connections.append(connection)

    def initialize_gene_randomly(self, number_of_nodes, under_one_threshold):

        node_list = []
        for i in range(number_of_nodes):
            random_weight = random.uniform(0, 1)
            first_node = Node(random_weight)
            node_list.append(first_node)

        # Add nodes to the gene's node list
        self.nodes = node_list

        for i in range(number_of_nodes):
            first_node = node_list[i]
            for j in range(number_of_nodes):
                second_node = node_list[j]
                random_connection_value = random.uniform(0, 1)
                if random_connection_value < under_one_threshold:
                    self.add_connection(first_node, second_node)

    def __str__(self):
        node_str = "\n  ".join(str(node) for node in self.nodes)
        connection_str = "\n  ".join(str(conn) for conn in self.connections)

        return f"Gene(\n  nodes=[\n  {node_str}\n  ],\n  connections=[\n  {connection_str}\n  ])"




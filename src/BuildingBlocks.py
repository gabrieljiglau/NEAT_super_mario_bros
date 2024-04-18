"""
Author: gabriel jiglau
StartingDate: 2024-III-26
Description: All the concepts I will be using, but fragmented into smaller parts
"""
import numpy as np


class NodesNotConnectedException(Exception):
    def __init__(self, first_node, second_node):
        self.message = f"Nodes {first_node} and {second_node} are not connected."

    def __str__(self):
        return self.message


"""
innovation_number :
i)a historical archive that keeps track of all the connections and nodes that have  been created across all generations;
ii)ensures consistency across different genomes, even if their topologies are different
"""


class Connection:
    def __init__(self, in_node, out_node, is_enabled, innovation_number):
        self.in_node = in_node
        self.out_node = out_node
        self.is_enabled = is_enabled
        self.innovation_number = innovation_number

    def __str__(self):
        return f"Connection from Node {self.in_node} to Node {self.out_node} is enabled : {self.is_enabled}, " \
               f"innovation_number: {self.innovation_number}"


class Node:
    def __init__(self, weight):
        # the weight should be set randomly at first
        self.weight = weight
        self.connections = []

    def __str__(self):
        return f"Node(weight={self.weight})"


""" 
the gene has the nodes and the connections 
it's the graph-like data-structure that holds them together
"""


class Gene:

    def __init__(self, nodes=None, connections=None):
        self.previous_innovation_numbers = []

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

    def add_node_between_genes(self, first_node: 'Node', second_node: 'Node', node_to_add: 'Node',
                               previous_innovation_numbers):

        if not self.are_nodes_connected(first_node, second_node):
            raise NodesNotConnectedException(first_node, second_node)

        # Check if the connection already exists in previous_innovation_numbers
        existing_innovation_number = self.find_matching_connection(first_node, second_node, previous_innovation_numbers)

        if existing_innovation_number:
            # Use the existing innovation_number
            is_enabled = True
            first_connection = Connection(first_node, node_to_add, is_enabled, existing_innovation_number)
            second_connection = Connection(node_to_add, second_node, is_enabled, existing_innovation_number)
        else:
            # Assign a new innovation_number
            is_enabled = True
            new_innovation_number = self.get_new_innovation_number()
            first_connection = Connection(first_node, node_to_add, is_enabled, new_innovation_number)
            second_connection = Connection(node_to_add, second_node, is_enabled, new_innovation_number)

        self.connections.append(first_connection)
        self.connections.append(second_connection)

        self.nodes.append(node_to_add)

        # Update the existing connections
        for connection in self.connections:
            if connection.out_node == second_node:
                connection.out_node = node_to_add
            if connection.in_node == second_node:
                connection.in_node = node_to_add

    @staticmethod
    def find_matching_connection(in_node: 'Node', out_node: 'Node', previous_innovation_numbers):
        for gene in previous_innovation_numbers:
            for connection in gene.connections:
                if connection.in_node == in_node and connection.out_node == out_node:
                    return connection
        return None

    def get_new_innovation_number(self):
        max_innovation = 0
        for connection in self.connections:
            max_innovation = max(max_innovation, connection.innovation_number)
        return max_innovation + 1

    """
    checks whether or not the connection exists in the topology;
    then adds the innovation_number accordingly(reuses it, or assigns a new value)
    """

    def add_connection(self, first_node: 'Node', second_node: 'Node'):
        existing_connection = self.find_matching_connection(first_node, second_node, self.previous_innovation_numbers)

        if existing_connection:
            connection = Connection(first_node, second_node, True, existing_connection.innovation_number)
        else:
            connection = Connection(first_node, second_node, True, self.get_new_innovation_number())

        self.connections.append(connection)

    """
    Check if a connection between two nodes already exists in previous_genes.
    i)if it does: reuse the existing connection's innovation number;
    ii)else: create a new connection with a new innovation number;
    """

    def initialize_gene_randomly(self, number_of_nodes, under_one_threshold, previous_innovation_numbers):

        node_list = []
        for i in range(number_of_nodes):
            random_weight = np.random.uniform(0, 1)
            first_node = Node(random_weight)
            node_list.append(first_node)

        # Add nodes to the gene's node list
        self.nodes = node_list

        for i in range(number_of_nodes):
            first_node = node_list[i]
            for j in range(number_of_nodes):
                second_node = node_list[j]
                random_connection_value = np.random.uniform(0, 1)

                if random_connection_value < under_one_threshold:
                    # Check if the connection already exists in previous_genes
                    existing_connection = self.find_matching_connection(first_node, second_node,
                                                                        previous_innovation_numbers)

                    if existing_connection:
                        # Use the existing innovation_number
                        connection = Connection(first_node, second_node, True, existing_connection.innovation_number)
                    else:
                        # Assign a new innovation_number
                        connection = Connection(first_node, second_node, True, self.get_new_innovation_number())

                    # Add the connection to the gene's connections list
                    self.connections.append(connection)

        # return instance of the class to allow method chaining
        return self

    def __str__(self):
        node_str = "\n  ".join(str(node) for node in self.nodes)
        connection_str = "\n  ".join(str(conn) for conn in self.connections)

        return f"Gene(\n  nodes=[\n  {node_str}\n  ],\n  connections=[\n  {connection_str}\n  ])"

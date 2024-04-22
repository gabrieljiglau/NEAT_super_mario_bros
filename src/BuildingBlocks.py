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


class Connection:
    def __init__(self, in_node_id: int, out_node_id: int, is_enabled: bool, innovation_number: int):
        self.in_node_id = in_node_id  # Index of the in_node in the gene's node list
        self.out_node_id = out_node_id  # Index of the out_node in the gene's node list
        self.is_enabled = is_enabled
        self.innovation_number = innovation_number

    def __str__(self):
        return f"Connection from Node {self.in_node_id} to Node {self.out_node_id} is enabled : {self.is_enabled}, " \
               f"innovation_number: {self.innovation_number}"


class Node:
    _id_counter = 0  # Class-level counter for assigning unique IDs to nodes

    def __init__(self, weight):
        self.id = Node._id_counter
        Node._id_counter += 1
        self.weight = weight
        self.connections = []

    def __str__(self):
        connections_str = ', '.join([str(conn) for conn in self.connections])
        return f"Node(id={self.id}, weight={self.weight}, connections=[{connections_str}])"

    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 0  # Reset the node ID counter for each new gene


"""
innovation_number :
i)a historical archive that keeps track of all the connections and nodes that have  been created across all generations;
ii)ensures consistency across different genomes, even if their topologies are different
"""

class InnovationCounter:
    def __init__(self):
        self.current_innovation_number = 0

    def get_new_innovation_number(self):
        self.current_innovation_number += 1
        return self.current_innovation_number


# global innovation counter
global_innovation_counter = InnovationCounter()

""" 
the gene has the nodes and the connections 
it's the graph-like data-structure that holds them together
"""


class Gene:
    _id_counter = 0  # class-level counter for assigning unique IDs to genes

    def __init__(self, nodes=None, connections=None):
        self.id = Gene._id_counter
        Gene._id_counter += 1  # Increment the counter for the next gene

        self.previous_innovation_numbers = {}

        # Reset the node ID counter for each new gene
        Node.reset_id_counter()

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes
        if connections is None:
            self.connections = []
        else:
            self.connections = connections

    def are_nodes_connected(self, first_node_id: int, second_node_id: int) -> bool:
        for connection in self.connections:
            if connection.in_node_id == first_node_id and connection.out_node_id == second_node_id:
                return True

        return False

    def is_connection_enabled(self, source_node_id: int, target_node_id: int) -> bool:
        for connection in self.connections:
            if connection.in_node_id == source_node_id and connection.out_node_id == target_node_id:
                return connection.is_enabled
        return False

    def add_node_between_genes(self, first_node_id: int, second_node_id: int, node_to_add: 'Node',
                               previous_innovation_numbers):

        if not self.are_nodes_connected(first_node_id, second_node_id):
            raise NodesNotConnectedException(first_node_id, second_node_id)

        innovation_number_first_connection = self.find_matching_connection(first_node_id, second_node_id,
                                                                           previous_innovation_numbers)
        innovation_number_second_connection = self.find_matching_connection(second_node_id, node_to_add.id,
                                                                            previous_innovation_numbers)
        is_enabled = True
        need_innovation_first = True
        need_innovation_second = True

        first_connection = None
        second_connection = None

        if innovation_number_first_connection:
            first_connection = Connection(first_node_id, node_to_add.id, is_enabled,
                                          innovation_number_first_connection.innovation_number)
            need_innovation_first = False

        if innovation_number_second_connection:
            second_connection = Connection(node_to_add.id, second_node_id, is_enabled,
                                           innovation_number_second_connection.innovation_number)
            need_innovation_second = False

        if need_innovation_first:
            new_innovation_number = self.get_new_innovation_number()
            first_connection = Connection(first_node_id, node_to_add.id, is_enabled, new_innovation_number)

        if need_innovation_second:
            new_innovation_number = self.get_new_innovation_number()
            second_connection = Connection(node_to_add.id, second_node_id, is_enabled, new_innovation_number)

        if first_connection is not None:
            self.connections.append(first_connection)

        if second_connection is not None:
            self.connections.append(second_connection)

        self.nodes.append(node_to_add)

        # Update the existing connections
        for connection in self.connections:
            if connection.out_node_id == second_node_id:
                connection.out_node_id = node_to_add.id
            if connection.in_node_id == second_node_id:
                connection.in_node_id = node_to_add.id

    @staticmethod
    def find_matching_connection(in_node_id: int, out_node_id: int, previous_innovation_numbers):
        innovation_key = (in_node_id, out_node_id)
        if innovation_key in previous_innovation_numbers:
            innovation_number = previous_innovation_numbers[innovation_key]
            return Connection(in_node_id, out_node_id, True, innovation_number)
        return None

    @staticmethod
    def get_new_innovation_number():
        return global_innovation_counter.get_new_innovation_number()

    def add_connection(self, first_node_id: int, second_node_id: int):
        innovation_key = (first_node_id, second_node_id)

        if innovation_key in self.previous_innovation_numbers:
            innovation_number = self.previous_innovation_numbers[innovation_key]
        else:
            new_innovation_number = self.get_new_innovation_number()
            innovation_number = new_innovation_number
            self.previous_innovation_numbers[innovation_key] = new_innovation_number

        connection = Connection(first_node_id, second_node_id, True, innovation_number)
        self.connections.append(connection)

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
            for j in range(i + 1, number_of_nodes):
                second_node = node_list[j]

                generated_threshold = np.random.uniform(0, 1)
                is_enabled = True

                if i != j and generated_threshold < under_one_threshold:
                    existing_connection = self.find_matching_connection(first_node.id, second_node.id,
                                                                        previous_innovation_numbers)

                    if existing_connection:
                        connection = Connection(first_node.id, second_node.id, is_enabled,
                                                existing_connection.innovation_number)
                    else:
                        new_innovation_number = self.get_new_innovation_number()
                        connection = Connection(first_node.id, second_node.id, is_enabled,
                                                new_innovation_number)

                        # Update previous_innovation_numbers with the new connection
                        innovation_key = (connection.in_node_id, connection.out_node_id)
                        previous_innovation_numbers[innovation_key] = connection.innovation_number

                    self.connections.append(connection)

        # return instance of the class to allow method chaining
        return self

    def __str__(self):
        node_str = "\n  ".join(str(node) for node in self.nodes)
        connection_str = "\n  ".join(str(conn) for conn in self.connections)

        return f"Gene(\n  nodes=[\n  {node_str}\n  ],\n  connections=[\n  {connection_str}\n  ])"

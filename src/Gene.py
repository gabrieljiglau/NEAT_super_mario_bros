"""
Author: gabriel jiglau
Date: 2024-III-26
Description: NEAT(neuro-evolution of augmenting topologies) for Geometry Dash
"""

class Connection:
    def __init__(self, in_node, out_node):
        self.in_node = in_node
        self.out_node = out_node

    def __str__(self):
        return f"Connection from Node {self.in_node} to Node {self.out_node}"

class Node:
    def __init__(self, weight, is_enabled, innovation_number):
        # the weight should be set randomly at first
        self.weight = weight
        self.is_enabled = is_enabled
        self.innovation_number = innovation_number
        self.connections = []

    def __str__(self):
        return f"Node(weight={self.weight}, is_enabled={self.is_enabled}, innovation_number={self.innovation_number})"

class Gene:
    def __init__(self, nodes, connections):
        self.nodes = nodes
        self.connections = connections

    def add_node_between_genes(self, first_node: 'Node', second_node: 'Node', node_to_add: 'Node'):
        # create new connections
        first_connection = Connection(first_node, node_to_add)
        second_connection = Connection(node_to_add, second_node)

        self.connections.append(first_connection)
        self.connections.append(second_connection)

        self.nodes.append(between_node)

    def add_connection(self, first_node: 'Node', second_node: 'Node'):
        # connection is bidirectional
        first_connection = Connection(first_node, second_node)
        second_connection = Connection(second_node, first_node)

        self.connections.append(first_connection)
        self.connections.append(second_connection)

    def are_nodes_connected(self, first_node: 'Node', second_node: 'Node') -> bool:
        # connection is bidirectional
        for connection in self.connections:
            if connection.in_node == first_node and connection.out_node == second_node:
                return True
            if connection.in_node == second_node and connection.out_node == first_node:
                return True
        return False

    def __str__(self):
        node_str = ", ".join(str(node) for node in self.nodes)
        connection_str = ", ".join(str(conn) for conn in self.connections)

        return f"Gene(nodes=[{node_str}], connections=[{connection_str}])"


def main():
        innovation_number = 1
        node_1 = Node(0.1, True, innovation_number)
        innovation_number += 1
        print(node_1)

        node_2 = Node(0.2, True, innovation_number)
        innovation_number += 1

        node_3 = Node(0.3, True, innovation_number)
        innovation_number += 1
        print(node_3)

        node_4 = Node(0.4, True, innovation_number)
        innovation_number += 1

        connection_1_to_4 = Connection(node_1, node_4)
        connection_2_to_4 = Connection(node_2, node_4)
        print(connection_1_to_4)
        print(connection_2_to_4)

        node_list = []
        node_list.append(node_1)
        node_list.append(node_2)
        node_list.append(node_3)
        node_list.append(node_4)

        connection_list = []
        connection_list.append(connection_1_to_4)
        connection_list.append(connection_2_to_4)

        chromosome = Gene(node_list, connection_list)
        print(chromosome.are_nodes_connected(node_1,node_4))

        # are_nodes_connected este testata; este buna

        # add_connection, respectiv add_node trebuie testate


if __name__ ==  "__main__":
        main()



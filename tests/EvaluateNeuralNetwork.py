from typing import List

from src.BuildingBlocks import Node, Connection
from src.NeuralNetwork import Gene

def create_nodes(number_of_nodes: int) -> List[Node]:

    nodes = []
    for i in range(number_of_nodes):
        if i < 4:
            bias = 1
        else:
            bias = 0

        new_node = Node(bias)
        nodes.append(new_node)

    return nodes

def create_connections(nodes: List[Node]) -> List[Connection]:
    connections = []

    n1_id = nodes[0].id
    n2_id = nodes[1].id
    n3_id = nodes[2].id
    n4_id = nodes[3].id

    n_in1 = nodes[4].id
    n_in2 = nodes[5].id

    n_out1 = nodes[6].id
    n_out2 = nodes[7].id
    n_out3 = nodes[8].id

    connection1 = Connection(n1_id, n3_id, 1.0, True, 1)
    connection2 = Connection(n1_id, n4_id, 1.0, True, 2)
    connection3 = Connection(n2_id, n3_id, 1.0, True, 3)
    connection4 = Connection(n2_id, n4_id, 1.0, True, 4)

    connections.append(connection1)
    connections.append(connection2)
    connections.append(connection3)
    connections.append(connection4)

    connection_in1 = Connection(n_in1, n1_id, -0.5, True, 5)
    connection_in1_prime = Connection(n_in1, n2_id, -0.5, True, 6)

    connection_in2 = Connection(n_in2, n1_id, -1, True, 7)
    connection_in2_prime = Connection(n_in2, n2_id, -1, True, 8)

    connections.append(connection_in1)
    connections.append(connection_in1_prime)
    connections.append(connection_in2)
    connections.append(connection_in2_prime)

    connection_out1 = Connection(n3_id, n_out1, 1, True, 9)
    connection_out1_prime = Connection(n3_id, n_out2, 1, True, 10)
    connection_out1_second = Connection(n3_id, n_out3, 1, True, 11)

    connections.append(connection_out1)
    connections.append(connection_out1_prime)
    connections.append(connection_out1_second)

    connection_out2 = Connection(n4_id, n_out1, 1, True, 12)
    connection_out2_prime = Connection(n4_id, n_out2, 1, True, 10)
    connection_out2_second = Connection(n4_id, n_out3, 1, True, 11)

    connections.append(connection_out2)
    connections.append(connection_out2_prime)
    connections.append(connection_out2_second)

    return connections

def create_network(number_of_nodes) -> Gene:
    node_list = create_nodes(number_of_nodes)
    connection_list = create_connections(number_of_nodes)

    return Gene(node_list, connection_list)

def simulate_input(neural_network: Gene):
    pass


if __name__ == '__main__':
    print('hi')
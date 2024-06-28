from typing import List

from src.BuildingBlocks import Node, Connection
from src.NeuralNetwork import Gene


def create_nodes(number_of_nodes: int) -> List[Node]:
    node_list = []
    for i in range(number_of_nodes):
        if i < 4:
            bias = 1
        else:
            bias = 0

        neighbours = []
        new_node = Node(neighbours, bias)
        node_list.append(new_node)

    return node_list

def add_connectivity_to_nodes(nodes_list: List[Node]) -> List[Node]:

    n1_id = nodes_list[0].id
    n2_id = nodes_list[1].id
    n3_id = nodes_list[2].id
    n4_id = nodes_list[3].id

    n_in1 = nodes_list[4].id
    n_in2 = nodes_list[5].id

    n_out1 = nodes_list[6].id
    n_out2 = nodes_list[7].id
    n_out3 = nodes_list[8].id

    connection1 = Connection(n1_id, n3_id, 1.0, True, 1)
    connection2 = Connection(n1_id, n4_id, 1.0, True, 2)
    connection3 = Connection(n2_id, n3_id, 1.0, True, 3)
    connection4 = Connection(n2_id, n4_id, 1.0, True, 4)

    nodes_list[0].add_connection_next_layer(nodes_list[2], connection1)
    nodes_list[0].add_connection_next_layer(nodes_list[3], connection2)

    nodes_list[1].add_connection_next_layer(nodes_list[2], connection3)
    nodes_list[1].add_connection_next_layer(nodes_list[3], connection4)

    connection_in1 = Connection(n_in1, n1_id, -0.5, True, 5)
    connection_in1_prime = Connection(n_in1, n2_id, -0.5, True, 6)

    connection_in2 = Connection(n_in2, n1_id, -1, True, 7)
    connection_in2_prime = Connection(n_in2, n2_id, -1, True, 8)

    # 4 and 5 are the input nodes
    nodes_list[4].add_connection_next_layer(nodes_list[0], connection_in1)
    nodes_list[4].add_connection_next_layer(nodes_list[1], connection_in1_prime)

    nodes_list[5].add_connection_next_layer(nodes_list[0], connection_in2)
    nodes_list[5].add_connection_next_layer(nodes_list[1], connection_in2_prime)

    # 6, 7 and 8 are the output nodes
    connection_out1 = Connection(n3_id, n_out1, 1, True, 9)
    connection_out1_prime = Connection(n3_id, n_out2, 1, True, 10)
    connection_out1_second = Connection(n3_id, n_out3, 1, True, 11)

    nodes_list[3].add_connection_next_layer(nodes_list[6], connection_out1)
    nodes_list[3].add_connection_next_layer(nodes_list[7], connection_out1_prime)
    nodes_list[3].add_connection_next_layer(nodes_list[8], connection_out1_second)

    connection_out2 = Connection(n4_id, n_out1, 1, True, 12)
    connection_out2_prime = Connection(n4_id, n_out2, 1, True, 10)
    connection_out2_second = Connection(n4_id, n_out3, 1, True, 11)

    nodes_list[4].add_connection_next_layer(nodes_list[6], connection_out2)
    nodes_list[4].add_connection_next_layer(nodes_list[7], connection_out2_prime)
    nodes_list[4].add_connection_next_layer(nodes_list[8], connection_out2_second)

    return nodes_list


"""
de mutat functia asta in 'Gene', care este, de fapt, o retea neuronala,
iar ea vi fi utilizata in functia de 'evaluare', cand individul 'se joaca' 'live'
"""


def process_input(network: Gene, input_value: float) -> float:
    output = 0

    for node in network.nodes:
        # call process_input_node in the Node class
        if node.is_input_neuron:
            # !! trebuie trecut ca parametru output doar daca nu este in stratul de input
            output += node.process_input_node(input_value)
        else:
            output += node.process_input_node(output)

    return output


if __name__ == '__main__':
    neurons = create_nodes(9)
    neurons = add_connectivity_to_nodes(neurons)

    neural_network = Gene(neurons)
    process_input(neural_network, 1)

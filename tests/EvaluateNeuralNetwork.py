from typing import List

from src.BuildingBlocks import Node, Connection
from src.NeuralNetwork import Gene


def create_nodes(input_neurons_number: int, hidden_neurons_number: List[int], output_neurons_number: int) -> List[Node]:
    nodes = []

    # Create input neurons
    for _ in range(input_neurons_number):
        nodes.append(Node(is_input_neuron=True))

    # Create hidden neurons
    for layer_size in hidden_neurons_number:
        for _ in range(layer_size):
            nodes.append(Node())

    # Create output neurons
    for _ in range(output_neurons_number):
        nodes.append(Node(is_output_neuron=True))

    return nodes


def connect_layers(from_layer: List[Node], to_layer: List[Node], innovation_number: int) -> List[Connection]:
    connections = []

    for from_node in from_layer:
        for to_node in to_layer:
            # Create a new connection
            connection = Connection(
                in_node_id=from_node.id,
                out_node_id=to_node.id,
                weight=1.0,  # default weight
                is_enabled=True,  # default enabled status
                innovation_number=innovation_number  # pass the innovation number
            )
            # Add the connection to the nodes
            from_node.add_connection_next_layer(to_node, connection)  # Add the connection for the from_node
            connections.append(connection)
            innovation_number += 1  # Increment for the next connection

    return connections


def create_network(num_input: int, num_hidden: List[int], num_output: int) -> Gene:
    global prev_layer
    nodes = create_nodes(num_input, num_hidden, num_output)
    connections = []

    input_layer = nodes[:num_input]
    current_index = num_input

    innovation_number = 1  # Initialize innovation number

    for layer_size in num_hidden:
        next_index = current_index + layer_size
        hidden_layer = nodes[current_index:next_index]
        connections.extend(connect_layers(input_layer if current_index == num_input else prev_layer,
                                          hidden_layer, innovation_number))
        prev_layer = hidden_layer
        current_index = next_index
        innovation_number += len(prev_layer) * len(hidden_layer)  # Increment by the number of connections added

    output_layer = nodes[current_index:]
    connections.extend(connect_layers(prev_layer, output_layer, innovation_number))

    return Gene(nodes, connections)

    return Gene(nodes, connections)

def process_input(network: Gene, input_value: float) -> List[float]:

    for node in network.nodes:
        if not node.is_visited:
            node.propagate_input(input_value)

    # Collect outputs from output neurons
    output_list = [node.output_value for node in network.nodes if node.is_output_neuron]

    return output_list


if __name__ == '__main__':
    num_input = 2
    num_hidden = [2, 2]
    num_output = 3

    neural_network = create_network(num_input, num_hidden, num_output)
    result_list = process_input(neural_network, 1)
    print('len of result_list = ', len(result_list))

    for i in range(len(result_list)):
        print('value : \n', result_list[i])


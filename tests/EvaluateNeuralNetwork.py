from typing import List

from src.BuildingBlocks import Node, Connection
from src.NeuralNetwork import Gene


def create_nodes(input_neurons_number: int, hidden_neurons_number: List[int], output_neurons_number: int) -> List[Node]:
    nodes = []

    # Create input neurons
    for i in range(input_neurons_number):
        nodes.append(Node(is_input_neuron=True))

    # Create hidden neurons
    hidden_layer_index = 0
    for layer_size in hidden_neurons_number:
        for j in range(layer_size):
            nodes.append(Node())
        hidden_layer_index += 1

    # Create output neurons
    for k in range(output_neurons_number):
        nodes.append(Node(is_output_neuron=True))

    return nodes


def connect_layers(from_layer: List[Node], to_layer: List[Node], innovation_number: int, weight: float) -> List[Connection]:
    connections = []

    for from_node in from_layer:
        for to_node in to_layer:
            # Create a new connection
            connection = Connection(
                in_node_id=from_node.id,
                out_node_id=to_node.id,
                weight=weight,  # use the specified weight
                is_enabled=True,  # default enabled status
                innovation_number=innovation_number  # pass the innovation number
            )
            # Add the connection to the nodes
            from_node.add_connection_next_layer(to_node, connection)  # Add the connection for the from_node
            connections.append(connection)
            innovation_number += 1  # Increment for the next connection

    return connections

def create_network(num_input: int, num_hidden: List[int], num_output: int) -> Gene:
    nodes = create_nodes(num_input, num_hidden, num_output)
    connections = []

    input_layer = nodes[:num_input]
    current_index = num_input

    innovation_number = 1  # Initialize innovation number

    # Connect input layer to first hidden layer with weights -0.5 and 0.5
    first_hidden_layer = nodes[current_index:current_index + num_hidden[0]]
    for i, from_node in enumerate(input_layer):
        weight = -0.5 if i % 2 == 0 else 0.5
        for to_node in first_hidden_layer:
            connection = Connection(
                in_node_id=from_node.id,
                out_node_id=to_node.id,
                weight=weight,
                is_enabled=True,
                innovation_number=innovation_number
            )
            from_node.add_connection_next_layer(to_node, connection)
            connections.append(connection)
            innovation_number += 1

    current_index += num_hidden[0]
    prev_layer = first_hidden_layer

    # Connect hidden layers with weight 1.0
    for layer_size in num_hidden[1:]:
        next_index = current_index + layer_size
        hidden_layer = nodes[current_index:next_index]
        connections.extend(connect_layers(prev_layer, hidden_layer, innovation_number, weight=1.0))
        prev_layer = hidden_layer
        current_index = next_index
        innovation_number += len(prev_layer) * layer_size  # Increment by the number of connections added

    # Connect last hidden layer to output layer with weight 0.5
    output_layer = nodes[current_index:]
    connections.extend(connect_layers(prev_layer, output_layer, innovation_number, weight=0.5))

    return Gene(nodes, connections)

# ceva e gresit cu calculul input-ului
def process_input(network: Gene, input_value: float) -> List[float]:

    for node in network.nodes:
        print('currently processing node with id: ',node.id)
        if not node.is_visited:
            node.process_input_node(input_value)

    # Collect outputs from output neurons
    output_list = [node.output_value for node in network.nodes if node.is_output_neuron]

    return output_list




if __name__ == '__main__':
    num_input = 2
    num_hidden = [2, 2]  # Two hidden layers with 2 neurons each
    num_output = 3

    neural_network = create_network(num_input, num_hidden, num_output)

    result_list = process_input(neural_network, 1.0)
    print("Output from network:", result_list)

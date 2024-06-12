from typing import List

from src.BuildingBlocks import Node, Connection

''' neutral network with 2 input and 3 output nodes'''


# Create nodes and connections
def create_network_with_four_nodes() -> List[Node]:
    nodes = [Node() for _ in range(9)]  # Add an extra node for the bias
    return nodes


def create_connections(nodes: List[Node]) -> List[Connection]:
    connections = []

    # Create connections between hidden nodes
    connection1 = Connection(nodes[0].id, nodes[2].id, 1.0, True, 1)
    connection2 = Connection(nodes[0].id, nodes[3].id, 1.0, True, 2)
    connection3 = Connection(nodes[1].id, nodes[2].id, 1.0, True, 3)
    connection4 = Connection(nodes[1].id, nodes[3].id, 1.0, True, 4)
    connections.extend([connection1, connection2, connection3, connection4])

    # Create connections from input nodes to hidden nodes
    connection_in1 = Connection(nodes[4].id, nodes[0].id, 0.5, True, 5)
    connection_in1_prime = Connection(nodes[4].id, nodes[1].id, 0.5, True, 6)
    connection_in2 = Connection(nodes[5].id, nodes[0].id, 0.5, True, 7)
    connection_in2_prime = Connection(nodes[5].id, nodes[1].id, 0.5, True, 8)
    connections.extend([connection_in1, connection_in1_prime, connection_in2, connection_in2_prime])

    # Create connections from hidden nodes to output nodes
    connection_out_11 = Connection(nodes[2].id, nodes[6].id, 1.0, True, 9)
    connection_out_12 = Connection(nodes[2].id, nodes[7].id, 1.0, True, 10)
    connection_out_13 = Connection(nodes[2].id, nodes[8].id, 1.0, True, 11)
    connections.extend([connection_out_11, connection_out_12, connection_out_13])

    connection_out_21 = Connection(nodes[3].id, nodes[6].id, 1.0, True, 12)
    connection_out_22 = Connection(nodes[3].id, nodes[7].id, 1.0, True, 13)
    connection_out_23 = Connection(nodes[3].id, nodes[8].id, 1.0, True, 14)
    connections.extend([connection_out_21, connection_out_22, connection_out_23])

    # Create connections from bias node to all hidden nodes
    bias_node_id = nodes[9].id  # Last node is the bias node
    connections.extend([
        Connection(bias_node_id, nodes[0].id, 0.5, True, 15),
        Connection(bias_node_id, nodes[1].id, 0.5, True, 16),
        Connection(bias_node_id, nodes[2].id, 0.5, True, 17),
        Connection(bias_node_id, nodes[3].id, 0.5, True, 18),
    ])

    return connections


# Forward propagation
def forward_propagation(nodes: List[Node], connections: List[Connection], input_values: List[float]):
    input_nodes = nodes[4:6]
    bias_node = nodes[9]
    for i, input_node in enumerate(input_nodes):
        input_node.input_value = input_values[i]
        input_node.output_value = input_values[i]

    bias_node.output_value = 1.0  # Bias node always outputs 1

    connection_map = {node.id: [] for node in nodes}
    for connection in connections:
        connection_map[connection.in_node_id].append(connection)

    for node in nodes:
        if node.id not in connection_map:
            continue

        for connection in connection_map[node.id]:
            if connection.is_enabled:
                out_node = next(n for n in nodes if n.id == connection.out_node_id)
                out_node.input_value += node.output_value * connection.weight

        node.activate()

    output_nodes = nodes[6:9]
    return [node.output_value for node in output_nodes]


# TODO: nodul de bias de adaugat in stratul de input

# probleme aici in main : IndexError: list index out of range
if __name__ == '__main__':
    print('hi')
    # Example usage
    nodes_list = create_network_with_four_nodes()
    connections_list = create_connections(nodes_list)
    input_values_list = [1.0, 1.0]  # Example input values
    output_values = forward_propagation(nodes_list, connections_list, input_values_list)

    print(f"Output values: {output_values}")

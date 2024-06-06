from typing import List

from src import NeuralNetwork

def create_four_nodes() -> List[NeuralNetwork.Node]:
    nodes = []

    node1 = NeuralNetwork.Node()
    node2 = NeuralNetwork.Node()
    node3 = NeuralNetwork.Node()
    node4 = NeuralNetwork.Node()

    nodes.append(node1)
    nodes.append(node2)
    nodes.append(node3)
    nodes.append(node4)

    node_in1 = NeuralNetwork.Node()
    node_in2 = NeuralNetwork.Node()

    nodes.append(node_in1)
    nodes.append(node_in2)

    node_out1 = NeuralNetwork.Node()
    node_out2 = NeuralNetwork.Node

    nodes.append(node_out1)
    nodes.append(node_out2)

    return nodes

# cum functioneaza innovation number acum daca iau in considerare si stratul de neuroni de input??
# eu zic ca nu ar trebui sa schimbe prea mult, deoarece input-ul nu face parte din topologie(??)
def create_connections(nodes: List[NeuralNetwork.Node]) -> List[NeuralNetwork.Connection]:
    connections = []

    n1_id = nodes[0].id
    n2_id = nodes[1].id
    n3_id = nodes[2].id
    n4_id = nodes[3].id

    n_in1 = nodes[4].id
    n_in2 = nodes[5].id

    n_out1 = nodes[6].id
    n_out2 = nodes[7].id

    connection1 = NeuralNetwork.Connection(n1_id, n3_id, 1.0, True, 1)
    connection2 = NeuralNetwork.Connection(n1_id, n4_id, 1.0, True, 2)
    connection3 = NeuralNetwork.Connection(n2_id, n3_id, 1.0, True, 3)
    connection4 = NeuralNetwork.Connection(n2_id, n4_id, 1.0, True, 4)

    connections.append(connection1)
    connections.append(connection2)
    connections.append(connection3)
    connections.append(connection4)

    connection_in1 = NeuralNetwork.Connection(n_in1, n1_id, 0.5, True, 5)
    connection_in1_prime = NeuralNetwork.Connection(n_in1, n2_id, True, 6)

    connection_in2 = NeuralNetwork.Connection(n_in2, n1_id, 0.5, True, 7)
    connection_in2_prime = NeuralNetwork.Connection(n_in2, n2_id, True, 8)

    connections.append(connection_in1)
    connections.append(connection_in1_prime)
    connections.append(connection_in2)
    connections.append(connection_in2_prime)

    # TODO : conexiunile de la nodurile 3 si 4 catre stratul de output

    return connections




if __name__ == '__main__':
    print('hi')
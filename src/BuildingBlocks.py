"""
Description: the concepts/building blocks that help build and manage the neural networks/genes
"""
from collections import deque
from typing import List

class NodesNotConnectedException(Exception):
    def __init__(self, first_node, second_node):
        self.message = f"Nodes {first_node} and {second_node} are not connected."

    def __str__(self):
        return self.message

class Connection:
    def __init__(self, in_node_id: int, out_node_id: int, weight: float, is_enabled: bool, innovation_number: int):
        self._in_node_id = in_node_id
        self._out_node_id = out_node_id
        self._weight = weight
        self._is_enabled = is_enabled
        self._innovation_number = innovation_number

    def __str__(self):
        return (f"Connection(in_node_id={self._in_node_id}, out_node_id={self._out_node_id}, "
                f"weight={self._weight}, is_enabled={self._is_enabled}, "
                f"innovation_number={self._innovation_number})")

    @property
    def out_node_id(self) -> int:
        return self._out_node_id

    @out_node_id.setter
    def out_node_id(self, out_node_id: int) -> None:
        self._out_node_id = out_node_id

    @property
    def in_node_id(self) -> int:
        return self._innovation_number

    @in_node_id.setter
    def in_node_id(self, in_node_id: int) -> None:
        self._in_node_id = in_node_id

    @property
    def innovation_number(self) -> int:
        return self._innovation_number

    @innovation_number.setter
    def innovation_number(self, new_innovation_number: int) -> None:
        self._innovation_number = new_innovation_number

    @property
    def is_enabled(self) -> bool:
        return self._is_enabled

    @is_enabled.setter
    def is_enabled(self, new_boolean: bool):
        self._is_enabled = new_boolean

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, new_weight: float = 0):
        self._weight = new_weight


# the node class now holds a reference to a list of nodes it is connected to
class Node:
    _id_counter = 0

    def __init__(self, neighbours=None, bias: float = 0, input_value: float = 0,
                 output_value: float = 0,
                 is_input_neuron: bool = False, is_output_neuron: bool = False):
        if neighbours is None:
            neighbours = []
        self.id = Node._id_counter
        Node._id_counter += 1

        self._neighbours = neighbours
        self.connections = []
        self._bias = bias
        self._input_value = input_value
        self._output_value = output_value
        self._is_input_neuron = is_input_neuron
        self._is_output_neuron = is_output_neuron
        self._is_visited = False

    def __str__(self):
        connections_str = ', '.join([str(conn) for conn in self.connections])
        return f"Node(id={self.id}, connections=[{connections_str}])"

    def process_input_node(self, input_value: float) -> None:
        if self.is_input_neuron:
            # for input neurons, pass the input unchanged to the first hidden layer
            self.output_value = input_value
            print('inner function processing node with id: ', self.id)
            print('Input nodes have output_value = ', self.output_value)

            for neighbour in self.neighbours:
                neighbour.input_value += self.output_value
                neighbour.input_value += self.bias
        else:
            # for hidden and output neurons, calculate the output value
            self.output_value = self.activate(self.input_value)
            print('Hidden/output nodes have output_value = ', self.output_value)
            print('Hidden/output function processing node with id: ', self.id)

            for i in range(len(self.neighbours)):
                current_neighbour = self.neighbours[i]
                current_connection = self.connections[i]
                if current_connection.is_enabled:
                    current_input = current_connection.weight * self.output_value + self.bias
                    current_neighbour.input_value += current_input

    def add_connection(self, connection: Connection):
        self.connections.append(connection)

    def add_connection_next_layer(self, node_in_next_layer: 'Node', connection: Connection):
        self._neighbours.append(node_in_next_layer)
        self.connections.append(connection)

    # ReLU's activation function
    def activate(self, current_sum: float) -> float:
        self._output_value = max(0.0, current_sum)
        return self._output_value

    # suspect method here !!
    @classmethod
    def _get_next_id(cls):
        result = cls._id_counter
        cls._id_counter += 1
        return result

    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 0

    @property
    def is_visited(self) -> bool:
        return self._is_visited

    @is_visited.setter
    def is_visited(self, is_visited: bool) -> None:
        self._is_visited = is_visited

    @property
    def bias(self) -> float:
        return self._bias

    @bias.setter
    def bias(self, new_bias: int = 0):
        self._bias = new_bias

    @property
    def neighbours(self) -> List['Node']:
        return self._neighbours

    @neighbours.setter
    def neighbours(self, neighbours: List['Node']):
        self._neighbours = neighbours

    @property
    def input_value(self) -> float:
        return self._input_value

    @input_value.setter
    def input_value(self, input_value: float = 0):
        self._input_value = input_value

    @property
    def output_value(self) -> float:
        return self._output_value

    @output_value.setter
    def output_value(self, output_value: float = 0):
        self._output_value = output_value

    @property
    def is_input_neuron(self) -> bool:
        return self._is_input_neuron

    @is_input_neuron.setter
    def is_input_neuron(self, is_input_neuron: bool):
        self._is_input_neuron = is_input_neuron

    @property
    def is_output_neuron(self) -> bool:
        return self._is_output_neuron

    @is_output_neuron.setter
    def is_output_neuron(self, is_output_neuron: bool):
        self._is_input_neuron = is_output_neuron


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

"""
Description: the concepts/building blocks that help build and manage the neural networks
"""


class Connection:
    def __init__(self, in_node_id: int, out_node_id: int, weight: float, is_enabled: bool, innovation_number: int):
        self.in_node_id = in_node_id
        self.out_node_id = out_node_id
        self.weight = weight
        self.is_enabled = is_enabled
        self.innovation_number = innovation_number

    def __str__(self):
        return (f"Connection(in_node_id={self.in_node_id}, out_node_id={self.out_node_id}, "
                f"weight={self.weight}, is_enabled={self.is_enabled}, "
                f"innovation_number={self.innovation_number})")

    @property
    def get_innovation_number(self):
        return self.innovation_number

    @property
    def is_connection_enabled(self):
        return self.is_enabled


class NodesNotConnectedException(Exception):
    def __init__(self, first_node, second_node):
        self.message = f"Nodes {first_node} and {second_node} are not connected."

    def __str__(self):
        return self.message

# de reparat Nodul astfel incat sa aibe in componenta si lista de noduri adiacente ??
# un exemplu:
'''
    private String name;
    private List<Vertex> neighbours;
    private List<Edge> edges;
    private boolean isVisited;

    public Vertex(String name){
        this.name = name;
        this.neighbours = new ArrayList<>();
        this.edges = new ArrayList<>();
    }

    public void addNeighbour(Vertex newVertex,Edge edge){
        neighbours.add(newVertex);
        edges.add(edge);
    }'''

class Node:
    _id_counter = 0

    def __init__(self, bias: float = 0, input_value: float = 0, output_value: float = 0,
                 is_input_neuron: bool = False, is_output_neuron: bool = False):
        self.id = Node._id_counter
        Node._id_counter += 1
        self.connections = []
        self._bias = bias
        self._input_value = input_value
        self._output_value = output_value
        self._is_input_neuron = is_input_neuron
        self._is_output_neuron = is_output_neuron

    def __str__(self):
        connections_str = ', '.join([str(conn) for conn in self.connections])
        return f"Node(id={self.id}, connections=[{connections_str}])"

    def add_connection(self, connection: Connection):
        self.connections.append(connection)

    # activate method should be private
    def _activate(self, current_sum: float) -> float:
        self._output_value = max(0.0, current_sum)  # ReLU's activation function
        return self._output_value

    # functia asta pare suspecta
    @classmethod
    def _get_next_id(cls):
        result = cls._id_counter
        cls._id_counter += 1
        return result

    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 0

    @property
    def bias(self) -> float:
        return self._bias

    @bias.setter
    def bias(self, new_bias: int = 0):
        self._bias = new_bias

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

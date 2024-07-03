"""
Author: gabriel jiglau
StartingDate: 2024-III-26
Description: The concepts regarding neural networks
"""
from typing import List
from collections import deque

import numpy as np
from src.BuildingBlocks import Connection, Node, NodesNotConnectedException, InnovationCounter

# global innovation counter
global_innovation_counter = InnovationCounter()

# global dictionary that keeps track of the innovation_numbers assigned to a connection between two nodes
previous_innovation_numbers = {}

"""
the gene has the nodes and the connections 
it's the graph-like data-structure that holds them together
"""


# TODO: add environment as a parameter in the constructor
class Gene:
    _id_counter = 0  # class-level counter for assigning unique IDs to genes

    # None este lasat aici doar pentru a testa simularea inputului din 'EvaluateNeuralNetwork.py'
    def __init__(self, nodes: List[Node], connections: List[Connection] = None, fitness_score: float = 0):
        self._fitness_score = fitness_score
        self.id = Gene._id_counter
        Gene._id_counter += 1  # Increment the counter for the next gene

        self.previous_innovation_numbers = {}

        # Reset the node ID counter for each new gene
        Node.reset_id_counter()

        if nodes is None:
            self._nodes = []
        else:
            self._nodes = nodes
        if connections is None:
            self.connections = []
        else:
            self.connections = connections

    """
         the idea is to pass the input_value to a specific node and 'do the math':
         i) sum the weighted inputs from all incoming connections
         ii) add the bias to this sum
         iii) apply the activation function

        then, propagate the result into the nodes that are connected to the 'current node'/the one passed as a parameter
        """

    # problema e aici; cred ca ar trebui sa procesezi current_node, si nu self,
    # adica sa ai current_node._process_input_node(input_value)

    # trebuie sa faci topological sort, pentru ca s-ar putea sa ai un nod care se coneteaza direct la un nod de output
    # A topological sort
    # may be performed on the nodes to find out in what order they should be selected for
    # passing on their internally stored values.

    def propagate_input(self, input_value: float) -> None:
        queue = deque()
        queue.append(self)

        while len(queue) > 0:
            current_node = queue.popleft()
            current_node.is_visited = True

            current_node.process_input_node(input_value)

            for neighbour in current_node.neighbours:
                if not neighbour.is_visited:
                    queue.append(neighbour)
                    neighbour.is_visited = True

    @property
    def nodes(self) -> List[Node]:
        return self._nodes

    @nodes.setter
    def nodes(self, nodes=List[Node] or None) -> None:
        self._nodes = nodes

    @property
    def get_connections(self):
        return self.connections

    @property
    def fitness_score(self) -> float:
        return self._fitness_score

    @fitness_score.setter
    def fitness_score(self, fitness_score: float):
        self._fitness_score = fitness_score

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

    def add_node_between_genes(self, first_node_id: int, second_node_id: int, node_to_add: 'Node'):

        if not self.are_nodes_connected(first_node_id, second_node_id):
            raise NodesNotConnectedException(first_node_id, second_node_id)

        # Disable the original connection and get its weight
        original_weight = None
        for connection in self.connections:
            if connection.in_node_id == first_node_id and connection.out_node_id == second_node_id:
                connection.is_enabled = False
                original_weight = connection.weight
                break

        if original_weight is None:
            raise NodesNotConnectedException(first_node_id, second_node_id)

        # Find or create innovation numbers for the new connections
        innovation_number_first_connection = self.find_matching_connection(first_node_id, node_to_add.id)
        innovation_number_second_connection = self.find_matching_connection(node_to_add.id, second_node_id)

        is_enabled = True

        if innovation_number_first_connection:
            first_connection = Connection(first_node_id, node_to_add.id, original_weight, is_enabled,
                                          innovation_number_first_connection.innovation_number)
        else:
            new_innovation_number = self.get_new_innovation_number()
            first_connection = Connection(first_node_id, node_to_add.id, original_weight, is_enabled,
                                          new_innovation_number)

        if innovation_number_second_connection:
            second_connection = Connection(node_to_add.id, second_node_id, original_weight, is_enabled,
                                           innovation_number_second_connection.innovation_number)
        else:
            new_innovation_number = self.get_new_innovation_number()
            second_connection = Connection(node_to_add.id, second_node_id, original_weight, is_enabled,
                                           new_innovation_number)

        # Add the new connections to the list
        self.connections.extend([first_connection, second_connection])

        # Add the new node to the nodes list
        self.nodes.append(node_to_add)

    # returns the innovation number of a connection if it exists, otherwise returns null
    # I have to get the connection weight and pass it as a parameter to the constructor, when returning it

    def find_matching_connection(self, in_node_id: int, out_node_id: int):
        innovation_key = (in_node_id, out_node_id)
        if innovation_key in previous_innovation_numbers:
            innovation_number = previous_innovation_numbers[innovation_key]
            # return Connection(in_node_id, out_node_id, True, innovation_number)
            for connection in self.connections:
                if connection.innovation_number == innovation_number:
                    return connection

        return None

    @staticmethod
    def get_new_innovation_number():
        return global_innovation_counter.get_new_innovation_number()

    def add_connection_between_nodes(self, first_node_id: int, second_node_id: int, weight: float):
        innovation_key = (first_node_id, second_node_id)

        if innovation_key in self.previous_innovation_numbers:
            innovation_number = self.previous_innovation_numbers[innovation_key]
        else:
            new_innovation_number = self.get_new_innovation_number()
            innovation_number = new_innovation_number
            self.previous_innovation_numbers[innovation_key] = new_innovation_number

        connection = Connection(first_node_id, second_node_id, weight, True, innovation_number)
        self.connections.append(connection)

    def initialize_gene_randomly(self, number_of_nodes, under_one_threshold):
        global previous_innovation_numbers, global_innovation_counter

        node_list = []
        for i in range(number_of_nodes):
            node_list.append(Node())

        self.nodes = node_list

        for i in range(number_of_nodes):
            for j in range(i + 1, number_of_nodes):
                generated_threshold = np.random.uniform(0, 1)

                if generated_threshold < under_one_threshold:
                    weight = np.random.uniform(-1, 1)  # Assign a random weight
                    is_enabled = True
                else:
                    continue

                if i == j:
                    continue

                existing_connection = self.find_matching_connection(node_list[i].id, node_list[j].id)

                if existing_connection:
                    connection = Connection(node_list[i].id, node_list[j].id, weight, is_enabled,
                                            existing_connection.innovation_number)
                else:
                    new_innovation_number = global_innovation_counter.get_new_innovation_number()
                    connection = Connection(node_list[i].id, node_list[j].id, weight, is_enabled, new_innovation_number)

                    innovation_key = (connection.in_node_id, connection.out_node_id)
                    previous_innovation_numbers[innovation_key] = connection.innovation_number

                self.connections.append(connection)

        return self

    def mutate_weights_gene(self, mutation_rate_weights: float, standard_deviation: float = 0.03):
        for connection in self.connections:
            generated_num = np.random.uniform(0, 1)
            if generated_num < mutation_rate_weights:
                connection.weight += np.random.normal(loc=0, scale=standard_deviation)

    # TODO: change the method, since now the node also has a field for 'neighbours',
    #  the list of nodes he is connected to
    def mutate_nodes_gene(self, mutation_rate_nodes: float):
        for connection in self.connections:
            generated_num = np.random.uniform(0, 1)

            if generated_num < mutation_rate_nodes:
                first_node_id = connection.in_node_id
                second_node_id = connection.out_node_id

                new_node = Node()
                # aici, noul nod, trebuie sa aibe ca vecin nodul numarul 2,
                # iar primul nod sa aibe ca vecin noul nod
                # TODO: inca nu m-am uitat in functie sa vad exact ce se intampla acolo
                self.add_node_between_genes(first_node_id, second_node_id, new_node)

    def mutate_connections_gene(self, mutation_rate_connections: float):
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if i != j and not self.are_nodes_connected(node1.id, node2.id):
                    generated_num = np.random.uniform(0, 1)
                    if generated_num < mutation_rate_connections:
                        weight = np.random.uniform(-1, 1)
                        self.add_connection_between_nodes(node1.id, node2.id, weight)

    def mutate_connections_disable_connection(self, mutation_rate_disable_connection: float):
        for connection in self.connections:
            generated_num = np.random.uniform(0, 1)
            if generated_num < mutation_rate_disable_connection is True:
                connection._is_enabled = False

    def mutate_connections_enable_connection(self, mutation_rate_enable_connection: float):
        for connection in self.connections:
            generated_num = np.random.uniform(0, 1)
            if generated_num < mutation_rate_enable_connection and connection.is_enabled is False:
                connection.is_enabled = True

    # TODO : interaction with the game
    # the objective fitness function
    def evaluate_individual(self, env, num_frames=50000):

        cumulative_reward = 0
        initial_x_position = 0
        initial_time = 400
        initial_score = 0
        initial_coins = 0
        done = True

        for frame in range(num_frames):
            if frame == 0 or done:
                env.reset()

            # here should be my actions that I take 'live'
            state, reward, done, info = env.step(env.action_space.sample())
            env.render()

            current_x_position = info['x_pos']
            velocity = current_x_position - initial_x_position

            current_time = info['time']
            clock_diff = initial_time - current_time

            current_score = info['score']
            score_diff = current_score - initial_score

            current_coins = info['coins']
            coins_diff = current_coins - initial_coins

            is_alive = info['life'] > 0
            death_penalty = -15 if not is_alive else 0

            reward = velocity + clock_diff + death_penalty + score_diff + coins_diff

            # Clip reward into range (-15, 15)
            reward = max(min(reward, 15), -15)

            # Update cumulative reward
            cumulative_reward += reward

            # update initial positions, time, score, and coins for the next iteration
            initial_x_position = current_x_position
            initial_time = current_time
            initial_score = current_score
            initial_coins = current_coins

            # check if the individual died or reached a terminal state
            if done:
                break

        return cumulative_reward

    def __str__(self):
        node_str = "\n  ".join(str(node) for node in self.nodes)
        connection_str = "\n  ".join(str(conn) for conn in self.connections)

        return f"Gene(\n  nodes=[\n  {node_str}\n  ],\n  connections=[\n  {connection_str}\n  ])"

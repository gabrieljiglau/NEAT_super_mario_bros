"""
Author: gabriel jiglau
StartingDate: 2024-III-26
Description: All the concepts I will be using, but fragmented into smaller parts
"""

import numpy as np


class NodesNotConnectedException(Exception):
    def __init__(self, first_node, second_node):
        self.message = f"Nodes {first_node} and {second_node} are not connected."

    def __str__(self):
        return self.message


class Connection:
    def __init__(self, in_node_id: int, out_node_id: int, is_enabled: bool, innovation_number: int):
        self.in_node_id = in_node_id
        self.out_node_id = out_node_id
        self.is_enabled = is_enabled
        self.innovation_number = innovation_number

    def __str__(self):
        return f"Connection from Node {self.in_node_id} to Node {self.out_node_id} is enabled : {self.is_enabled}, " \
               f"innovation_number: {self.innovation_number}"

    @property
    def get_innovation_number(self):
        return self.innovation_number

    @property
    def is_connection_enabled(self):
        return self.is_enabled


class Node:
    _id_counter = 0  # class-level counter for assigning unique IDs to nodes

    def __init__(self, weight):
        self.id = Node._id_counter
        Node._id_counter += 1
        self.weight = weight
        self.connections = []

    def __str__(self):
        connections_str = ', '.join([str(conn) for conn in self.connections])
        return f"Node(id={self.id}, weight={self.weight}, connections=[{connections_str}])"

    @classmethod
    def reset_id_counter(cls):
        cls._id_counter = 0  # reset the node ID counter for each new gene


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

    def __init__(self, nodes=None, connections=None, fitness_score: int = 0):
        self.fitness_score = fitness_score
        self.id = Gene._id_counter
        Gene._id_counter += 1  # Increment the counter for the next gene

        self.previous_innovation_numbers = {}

        # Reset the node ID counter for each new gene
        Node.reset_id_counter()

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes
        if connections is None:
            self.connections = []
        else:
            self.connections = connections

    @property
    def get_fitness_score(self):
        return self.fitness_score

    def set_fitness_score(self, fitness_score):
        self.fitness_score = fitness_score

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

        # disable the original connection
        for connection in self.connections:
            if connection.out_node_id == second_node_id:
                connection.is_enabled = False

        innovation_number_first_connection = self.find_matching_connection(first_node_id, second_node_id)
        innovation_number_second_connection = self.find_matching_connection(second_node_id, node_to_add.id)

        is_enabled = True
        need_innovation_first = True
        need_innovation_second = True

        first_connection = None
        second_connection = None

        # in Python, any non-zero number is considered True, and zero is considered False.
        if innovation_number_first_connection:
            first_connection = Connection(first_node_id, node_to_add.id, is_enabled,
                                          innovation_number_first_connection.innovation_number)
            need_innovation_first = False

        if innovation_number_second_connection:
            second_connection = Connection(node_to_add.id, second_node_id, is_enabled,
                                           innovation_number_second_connection.innovation_number)
            need_innovation_second = False

        if need_innovation_first:
            new_innovation_number = self.get_new_innovation_number()
            first_connection = Connection(first_node_id, node_to_add.id, is_enabled, new_innovation_number)

        if need_innovation_second:
            new_innovation_number = self.get_new_innovation_number()
            second_connection = Connection(node_to_add.id, second_node_id, is_enabled, new_innovation_number)

        if first_connection is not None:
            self.connections.append(first_connection)

        if second_connection is not None:
            self.connections.append(second_connection)

        # add the new node to the nodes list
        self.nodes.append(node_to_add)

    # returns the innovation number of a connection if it exists, otherwise returns null
    @staticmethod
    def find_matching_connection(in_node_id: int, out_node_id: int):
        innovation_key = (in_node_id, out_node_id)
        if innovation_key in previous_innovation_numbers:
            innovation_number = previous_innovation_numbers[innovation_key]
            return Connection(in_node_id, out_node_id, True, innovation_number)
        return None

    @staticmethod
    def get_new_innovation_number():
        return global_innovation_counter.get_new_innovation_number()

    def add_connection_between_nodes(self, first_node_id: int, second_node_id: int):
        innovation_key = (first_node_id, second_node_id)

        if innovation_key in self.previous_innovation_numbers:
            innovation_number = self.previous_innovation_numbers[innovation_key]
        else:
            new_innovation_number = self.get_new_innovation_number()
            innovation_number = new_innovation_number
            self.previous_innovation_numbers[innovation_key] = new_innovation_number

        connection = Connection(first_node_id, second_node_id, True, innovation_number)
        self.connections.append(connection)

    def initialize_gene_randomly(self, number_of_nodes, under_one_threshold):
        global previous_innovation_numbers, global_innovation_counter

        node_list = []
        for i in range(number_of_nodes):
            random_weight = np.random.uniform(0, 1)
            first_node = Node(random_weight)
            node_list.append(first_node)

        # Add nodes to the gene's node list
        self.nodes = node_list

        # old double for loop that didn't add mark the connections under that threshold as having 'is_enabled' = false
        # and didn't add the 'dormant' connection in the list of connections, but worked
        '''
        for i in range(number_of_nodes):
            first_node = node_list[i]
            for j in range(i + 1, number_of_nodes):
                second_node = node_list[j]

                generated_threshold = np.random.uniform(0, 1)
                is_enabled = True
                
                # want to add a connection if only the generated value is under a given threshold
                if i != j and generated_threshold < under_one_threshold:
                    existing_connection = self.find_matching_connection(first_node.id, second_node.id)

                    if existing_connection:
                        connection = Connection(first_node.id, second_node.id, is_enabled,
                                                existing_connection.innovation_number)
                    else:
                        new_innovation_number = global_innovation_counter.get_new_innovation_number()
                        connection = Connection(first_node.id, second_node.id, is_enabled,
                                                new_innovation_number)

                        # Update previous_innovation_numbers with the new connection
                        innovation_key = (connection.in_node_id, connection.out_node_id)
                        previous_innovation_numbers[innovation_key] = connection.innovation_number

                    self.connections.append(connection)
        '''
        # updated version, but untested
        for i in range(number_of_nodes):
            first_node = node_list[i]
            for j in range(i + 1, number_of_nodes):
                second_node = node_list[j]

                # Generate a random value
                generated_threshold = np.random.uniform(0, 1)

                # Check if the generated value is under the threshold
                if generated_threshold < under_one_threshold:
                    is_enabled = True  # Enable the connection
                else:
                    is_enabled = False  # Disable the connection

                # Check if the nodes are the same
                if i == j:
                    continue  # Skip the rest of the loop iteration if the nodes are the same

                # Find or create a connection
                existing_connection = self.find_matching_connection(first_node.id, second_node.id)

                # what happens on this line when the connection was added since the first to the second node
                if existing_connection:
                    connection = Connection(first_node.id, second_node.id, is_enabled,
                                            existing_connection.innovation_number)
                else:
                    new_innovation_number = global_innovation_counter.get_new_innovation_number()
                    connection = Connection(first_node.id, second_node.id, is_enabled, new_innovation_number)

                    # Update previous_innovation_numbers with the new connection
                    innovation_key = (connection.in_node_id, connection.out_node_id)
                    previous_innovation_numbers[innovation_key] = connection.innovation_number

                # Add the connection to the list
                self.connections.append(connection)

        # return instance of the class to allow method chaining
        return self

    def mutate_weights_gene(self, mutation_rate_weights: float, standard_deviation: float = 0.03):
        for node in self.nodes:
            generated_num = np.random.uniform(0, 1)
            if generated_num < mutation_rate_weights:
                # add a random number sampled from a normal distribution with
                # a mean of 0 and a standard deviation of standard_deviation
                node.weight += np.random.normal(loc=0, scale=standard_deviation)

    def mutate_nodes_gene(self, mutation_rate_nodes: float):
        for connection in self.connections:
            generated_num = np.random.uniform(0, 1)

            if generated_num < mutation_rate_nodes:
                first_node_id = connection.in_node_id
                second_node_id = connection.out_node_id

                generated_weight = np.random.uniform(0, 1)
                new_node = Node(generated_weight)

                self.add_node_between_genes(first_node_id, second_node_id, new_node)

    def mutate_connections_gene(self, mutation_rate_connections: float):
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if i != j and not self.are_nodes_connected(node1.id, node2.id):
                    generated_num = np.random.uniform(0, 1)
                    if generated_num < mutation_rate_connections:
                        # Add a connection between node1 and node2
                        self.add_connection_between_nodes(node1.id, node2.id)

    def mutate_connections_disable_connection(self, mutation_rate_disable_connection: float):
        for connection in self.connections:
            generated_num = np.random.uniform(0, 1)
            if generated_num < mutation_rate_disable_connection is True:
                connection.is_enabled = False

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

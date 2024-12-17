
class MetaGeneticAlgorithm:

    def __init__(self, pop_size: int, crossover_rate: float, crossover_points: int, mutation_rate_discrete: float,
                 mutation_rate_continuous: float, max_generations: int = 350):

        self.pop_size = pop_size
        self.crossover_rate = crossover_rate
        self.crossover_points = crossover_points
        self.mutation_rate_discrete = mutation_rate_discrete
        self.mutation_rate_continuous = mutation_rate_continuous
        self.number_of_elite = int(pop_size * 0.085)
        self.max_generations = max_generations


    def optimize_network_hyperparameters(self):
        pass


if __name__ == '__main__':
    pass

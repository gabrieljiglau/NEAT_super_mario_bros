import unittest
from tests.edit_neat_configuration import run_neat

class TestNEATConfig(unittest.TestCase):

    @staticmethod
    def run_multiple_generations(config_file, generations, config_updates, runs):
        fitness_scores = []
        for _ in range(runs):
            stats = run_neat(config_file, generations, config_updates)
            fitness = max(c.fitness for c in stats.best_genomes(-1))  # Get the best fitness from the last generation
            fitness_scores.append(fitness)
        return sum(fitness_scores) / len(fitness_scores)  # Return average fitness

    def test_bias_mutate_rate(self):
        config_file = 'config'
        generations = 10
        runs = 10  # Number of times to run NEAT for averaging

        config_updates_0_5 = {'bias_mutate_rate': 0.5}
        fitness_0_5 = self.run_multiple_generations(config_file, generations, config_updates_0_5, runs)

        # Run with bias_mutate_rate = 0.7
        config_updates_0_7 = {'bias_mutate_rate': 0.7}
        fitness_0_7 = self.run_multiple_generations(config_file, generations, config_updates_0_7, runs)

        print(f"Average fitness with bias_mutate_rate 0.5: {fitness_0_5}")
        print(f"Average fitness with bias_mutate_rate 0.7: {fitness_0_7}")

        # Assert that the lower bias_mutate_rate (0.5) yields better results
        self.assertGreater(fitness_0_5, fitness_0_7, "bias_mutate_rate 0.5 should perform better than 0.7")

    def test_enable_mutate_rate(self):
        config_file = 'config'
        generations = 10
        runs = 10  # Number of times to run NEAT for averaging

        standard_config = {'enabled_mutate_rate': 0.05}
        standard_fitness = self.run_multiple_generations(config_file, generations, standard_config, runs)

        updated_config = {'enabled_mutate_rate': 0.95}
        updated_fitness = self.run_multiple_generations(config_file, generations, updated_config, runs)

        print(f"Average fitness with enabled_mutate_rate 0.05: {standard_fitness}")
        print(f"Average fitness with enabled_mutate_rate 0.95: {updated_fitness}")

        # Assert that the lower bias_mutate_rate (0.5) yields better results
        self.assertGreater(standard_fitness, updated_fitness, "enable_mutate_rate 0.05 should perform better than 0.95")

    def test_hidden_neurons(self):
        config_file = 'config'  # Adjust path to your actual config file
        generations = 10
        runs = 10  # Number of times to run NEAT for averaging

        # Run with num_hidden = 0
        config_updates_0 = {'num_hidden': 0}
        fitness_0 = self.run_multiple_generations(config_file, generations, config_updates_0, runs)

        # Run with num_hidden = 4
        config_updates_4 = {'num_hidden': 4}
        fitness_4 = self.run_multiple_generations(config_file, generations, config_updates_4, runs)

        print(f"Average fitness with num_hidden 0: {fitness_0}")
        print(f"Average fitness with num_hidden 4: {fitness_4}")

        # Assert that having hidden neurons (num_hidden = 4) yields better results
        self.assertGreater(fitness_4, fitness_0, "num_hidden 4 should perform better than 0")

    def test_survival_threshold(self):
        config_file = 'config'
        generations = 10
        runs = 10  # Number of times to run NEAT for averaging

        # Run with survival_threshold = 0.2
        config_updates_0_2 = {'survival_threshold': 0.2}
        fitness_0_2 = self.run_multiple_generations(config_file, generations, config_updates_0_2, runs)

        # Run with survival_threshold = 0.7
        config_updates_0_7 = {'survival_threshold': 0.7}
        fitness_0_7 = self.run_multiple_generations(config_file, generations, config_updates_0_7, runs)

        print(f"Average fitness with survival_threshold 0.2: {fitness_0_2}")
        print(f"Average fitness with survival_threshold 0.7: {fitness_0_7}")

        # Assert that the lower survival_threshold (0.2) yields better results
        self.assertGreater(fitness_0_7, fitness_0_2, "survival_threshold 0.7 should perform better than 0.2")

    # testul asta este cazatura, ceva nu este ok
    def test_initial_connection(self):
        config_file = 'config'
        generations = 10
        runs = 10  # Number of times to run NEAT for averaging

        # Run with survival_threshold = 0.2
        config_updates_partial_connectivity = {'initial_connection': 'partial_connected'}
        partial_connectivity_result = self.run_multiple_generations(config_file, generations,
                                                                    config_updates_partial_connectivity, runs)

        # Run with survival_threshold = 0.7
        config_updates_full_connectivity = {'initial_connection': 'full'}
        full_connectivity_result = self.run_multiple_generations(config_file, generations,
                                                                 config_updates_full_connectivity, runs)

        print(f"Average fitness with survival_threshold 0.2: {partial_connectivity_result}")
        print(f"Average fitness with survival_threshold 0.7: {full_connectivity_result}")

        # Assert that the lower survival_threshold (0.2) yields better results
        self.assertGreater(partial_connectivity_result, full_connectivity_result,
                           "initial_connection partial_connection should perform better than full")


if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestNEATConfig('test_hidden_neurons'))

    # Run the test suite
    runner = unittest.TextTestRunner()
    runner.run(suite)

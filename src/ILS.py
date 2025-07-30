from configparser import ConfigParser
import copy
from algorithmBase import AlgorithmBase, AlgorithmConfigBase
import random


class ILSConfig(AlgorithmConfigBase):
    def __init__(self, config: ConfigParser):
        self.pertubation_factor = 0.1
        self.neighbor_size = 10
        self.hamming_distance = 1
        super().__init__(config)


# all parameter values are bound between 0 and 1, later to be expanded:
BOUNDS_LOW, BOUNDS_HIGH = 0.0, 1.0  # boundaries for all dimensions


class ILS(AlgorithmBase):
    def __init__(self, config: AlgorithmConfigBase, image_file: str, output_folder: str, id: str):
        super().__init__(config, image_file, output_folder, id)

    def perturbation(self, elem, delta):
        m = (BOUNDS_HIGH-BOUNDS_LOW)*delta
        perturbed = [old+random.uniform(-m, m) for old in elem]
        return perturbed

    def mutation(self, elem, n=1):
        mutated = copy.copy(elem)
        for _ in range(n):
            idx = random.randint(0, self.num_of_params-1)
            mutated[idx] = random.uniform(BOUNDS_LOW, BOUNDS_HIGH)
        return mutated

    def executive(self):
        config: ILSConfig = self.config

        self._beginExecution()

        # create a random solution
        best_solution = self.randomSolution()
        best_fitness = self.objectiveFunction(best_solution)
        self._updateExecution(best_fitness, best_solution)

        while self._isExecutable():

            new_solution = self.perturbation(
                best_solution, config.pertubation_factor)
            new_solution_fitness = self.objectiveFunction(new_solution)

            for _ in range(config.neighbor_size):
                close_solution = self.mutation(
                    new_solution, config.hamming_distance)
                close_solution_fitness = self.objectiveFunction(close_solution)

                if close_solution_fitness < new_solution_fitness:
                    new_solution_fitness = close_solution_fitness
                    new_solution = close_solution

            fitness_improved = False
            if new_solution_fitness <= best_fitness:
                best_solution = new_solution
                best_fitness = new_solution_fitness
                fitness_improved = True

            delta = 0
            if self.config.save_image_each == -1 and fitness_improved:
                delta = delta+self.saveImage(str(self.currentGen),
                                             best_solution)
            elif self.config.save_image_each >= 1 and self.currentGen % self.config.save_image_each == 0:
                delta = delta+self.saveImage(str(self.currentGen),
                                             best_solution)
            self._updateExecution(best_fitness,
                                  best_solution,
                                  delta=delta)

        self._endExecution()
        return best_solution

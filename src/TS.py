from configparser import ConfigParser
from algorithmBase import AlgorithmBase, AlgorithmConfigBase
import random


class TSConfig(AlgorithmConfigBase):
    def __init__(self, config: ConfigParser):
        self.pertubation_factor = 0.1
        self.tabu_list_size = 10
        super().__init__(config)


# all parameter values are bound between 0 and 1, later to be expanded:
BOUNDS_LOW, BOUNDS_HIGH = 0.0, 1.0  # boundaries for all dimensions


class TS(AlgorithmBase):
    def __init__(self, config: AlgorithmConfigBase, image_file: str, output_folder: str, id: str):
        super().__init__(config, image_file, output_folder, id)

    def perturbation(self, elem, delta):
        m = (BOUNDS_HIGH-BOUNDS_LOW)*delta
        perturbed = [old+random.uniform(-m, m) for old in elem]
        return perturbed

    def executive(self):
        config: TSConfig = self.config

        self._beginExecution()

        # Initialize the best solution and its value
        best_solution = None
        best_value = float('inf')

        # Initialize the tabu list
        tabu_list = []

        # Initialize the current solution and its value
        current_solution = self.randomSolution()
        current_value = self.objectiveFunction(current_solution)
        self._updateExecution(current_value, current_solution)

        # Iterate for the specified number of iterations
        while self._isExecutable():
            fitness_improved = False

            # If the current solution is better than the best solution, update the best solution
            if current_value < best_value:
                best_solution = current_solution
                best_value = current_value
                fitness_improved = True

            # Add the current solution to the tabu list
            if not current_solution in tabu_list:
                tabu_list.append(current_solution)

            # Generate a new solution by perturbing the current solution
            new_solution = self.perturbation(
                current_solution, config.pertubation_factor)

            # Evaluate the new solution
            new_value = self.objectiveFunction(new_solution)

            # If the new solution is better than the current solution, update the current solution
            if new_value < current_value:
                current_solution = new_solution
                current_value = new_value
            else:
                # If the new solution is not better than the current solution, check if it is in the tabu list
                if new_solution in tabu_list:
                    # If it is, generate a new solution by perturbing the current solution
                    current_solution = self.perturbation(
                        current_solution, config.pertubation_factor)
                    current_value = self.objectiveFunction(new_solution)
                else:
                    # If it is not, add the new solution to the tabu list
                    tabu_list.append(new_solution)

            # If the tabu list is full, remove the oldest solution from it
            if len(tabu_list) > config.tabu_list_size:
                for _ in range(len(tabu_list)-config.tabu_list_size):
                    tabu_list.pop(0)

            # Print the current best solution and its value
            delta = 0
            if self.config.save_image_each == -1 and fitness_improved:
                delta = delta+self.saveImage(str(self.currentGen),
                                             best_solution)
            elif self.config.save_image_each >= 1 and self.currentGen % self.config.save_image_each == 0:
                delta = delta+self.saveImage(str(self.currentGen),
                                             best_solution)
            self._updateExecution(best_value,
                                  best_solution,
                                  delta=delta)

        self._endExecution()
        return best_solution

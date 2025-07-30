from algorithmBase import BOUNDS_HIGH, BOUNDS_LOW, AlgorithmBase, AlgorithmConfigBase
from configparser import ConfigParser
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import numpy as np
import random


class GAConfig(AlgorithmConfigBase):
    def __init__(self, config: ConfigParser):
        self.population_size = 200
        self.prob_crossover = 0.9
        self.prob_mutation = 0.5
        self.hall_of_fame_size = 20
        self.crowding_factor = 10.0  # crowding factor for crossover and mutation
        super().__init__(config)


class GA(AlgorithmBase):
    def __init__(self, config: AlgorithmConfigBase, image_file: str, output_folder: str, id: str):
        super().__init__(config, image_file, output_folder, id)

    def __getToolbox(self, num_of_params, crowding_factor):
        toolbox = base.Toolbox()

        # define a single objective, minimizing fitness strategy:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

        # create the Individual class based on list:
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # create an operator that randomly returns a float in the desired range:
        toolbox.register("attrFloat", self.randomSolution)

        # create an operator that fills up an Individual instance:
        toolbox.register("individualCreator",
                         tools.initIterate,
                         creator.Individual,
                         toolbox.attrFloat)

        # create an operator that generates a list of individuals:
        toolbox.register("populationCreator",
                         tools.initRepeat,
                         list,
                         toolbox.individualCreator)

        # fitness calculation
        toolbox.register("evaluate", self.objectiveFunction)

        # genetic operators:
        toolbox.register("select", tools.selTournament, tournsize=2)

        toolbox.register("mate",
                         tools.cxSimulatedBinaryBounded,
                         low=BOUNDS_LOW,
                         up=BOUNDS_HIGH,
                         eta=crowding_factor)

        toolbox.register("mutate",
                         tools.mutPolynomialBounded,
                         low=BOUNDS_LOW,
                         up=BOUNDS_HIGH,
                         eta=crowding_factor,
                         indpb=1.0/num_of_params)
        return toolbox

    def executive(self):
        """This algorithm is similar to DEAP eaSimple() algorithm, with two additions:
        1. halloffame is used to implement an elitism mechanism. The individuals contained in the
        halloffame are directly injected into the next generation and are not subject to the
        genetic operators of selection, crossover and mutation.
        2. a callback argument was added. It represents an external function that will be called after
        each iteration, passing the current generation number and the current best individual as arguments
        """

        config: GAConfig = self.config
        ngen = config.max_generation
        cxpb = config.prob_crossover
        mutpb = config.prob_mutation

        toolbox = self.__getToolbox(self.num_of_params, config.crowding_factor)
        # define the hall-of-fame object:
        halloffame = tools.HallOfFame(config.hall_of_fame_size)

        self._beginExecution()

        # create initial population (generation 0):
        population = toolbox.populationCreator(n=config.population_size)

        def getCurrentBest(pop):
            fitness_best = None
            value_best = None
            fitness_list = []

            for item in pop:
                fit = item.fitness.getValues()[0]
                fitness_list.append(fit)
                if fitness_best == None or fit < fitness_best:
                    fitness_best = fit
                    value_best = item

            fitness_list = np.array(fitness_list)
            return fitness_best, value_best, fitness_list.max(), fitness_list.mean(), fitness_list.std()

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit,

        if halloffame is None:
            raise ValueError("halloffame parameter must not be empty!")

        halloffame.update(population)
        hof_size = len(halloffame.items) if halloffame.items else 0

        best_fitness, best_solution, worse_solution, mean_solution, std_solution = getCurrentBest(
            population)
        self._updateExecution(best_fitness,
                              best_solution,
                              worse_solution,
                              mean_solution,
                              std_solution)

        # Begin the generational process
        while self._isExecutable():

            # Select the next generation individuals
            offspring = toolbox.select(population, len(population) - hof_size)

            # Vary the pool of individuals
            offspring = algorithms.varAnd(
                offspring, toolbox, cxpb, mutpb)

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit,

            # add the best back to population:
            offspring.extend(halloffame.items)

            # Update the hall of fame with the generated individuals
            halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            current_fitness, current_solution, current_fitness_worse, current_fitness_mean, current_fitness_std = getCurrentBest(
                population)
            fitness_improved = False
            if current_fitness < best_fitness:
                best_fitness = current_fitness
                best_solution = current_solution
                fitness_improved = True

            delta = 0
            if self.config.save_image_each == -1 and fitness_improved:
                delta = delta+self.saveImage(str(self.currentGen),
                                             best_solution)
            elif self.config.save_image_each >= 1 and self.currentGen % self.config.save_image_each == 0:
                delta = delta+self.saveImage(str(self.currentGen),
                                             best_solution)

            self._updateExecution(current_fitness,
                                  current_solution,
                                  current_fitness_worse,
                                  current_fitness_mean,
                                  current_fitness_std,
                                  delta=delta)

        self._endExecution()

        best = halloffame.items[0]
        return best

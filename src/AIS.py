from configparser import ConfigParser
from algorithmBase import AlgorithmBase, AlgorithmConfigBase
import math
from random import random
from copy import deepcopy


class AISConfig(AlgorithmConfigBase):
    def __init__(self, config: ConfigParser):
        # Initial number of antibodies.
        self.number_of_antibodies = 100
        # The clonation's occurrence rate.
        self.clone_rate = 0.1
        # Mutation exponent, mutation probability is proportional to exp(-affinity*mutation_exp).
        self.mutation_exp = 0.4
        # The maximum number of antibodies that are kept between subsequent iterations.
        self. max_antibodies = 100
        # The number of the worst antibodies that are removed on each iteration.
        self.num_remove = 2
        # The number of antibodies being returned.
        self.mem_size = 20
        super().__init__(config)

# Clonal Selection Algorithm (Artificial Immune System)


class AISAntibody:
    paratopes: list = []
    affinity: float = None

    def clone(self):
        a = AISAntibody()
        a.paratopes = self.paratopes.copy()
        a.affinity = self.affinity
        return a


class AIS(AlgorithmBase):
    def __init__(self, config: AlgorithmConfigBase, image_file: str, output_folder: str, id: str):
        super().__init__(config, image_file, output_folder, id)

    def random_antibody_fcn(self):
        antibody = AISAntibody()
        antibody.paratopes = self.randomSolution()
        antibody.affinity = None
        return antibody

    def calculate_affinity_fcn(self, antibodies):
        for x in antibodies:
            antibody: AISAntibody = x
            antibody.affinity = -self.objectiveFunction(antibody.paratopes)

    def clone_antibodies_fcn(self, antibodies, clone_rate):
        clones = []
        affinities = [x.affinity for x in antibodies]
        max_affinity = max(affinities)
        min_affinity = min(affinities)
        for x in antibodies:
            a: AISAntibody = x
            n_clone = int(math.ceil(len(antibodies) * ((a.affinity -
                          min_affinity)/(max_affinity-min_affinity)) * clone_rate))
            clones += [a.clone() for _ in range(n_clone)]
        #for c in clones:
        #    c.affinity = 0
        return clones

    def mutation_fcn(self, clones, mutation_exp):
        affinities = [x.affinity for x in clones]
        max_affinity = max(affinities)
        min_affinity = min(affinities)

        for i in range(0, len(clones)):
            a: AISAntibody = clones[i]
            aff = ((a.affinity - min_affinity)/(max_affinity-min_affinity))
            mutation_rate = math.exp(-aff * mutation_exp)
            clones[i] = self.point_mutation(clones[i], mutation_rate)
        return clones

    def selection_fcn(self, antibodies):
        return sorted(antibodies, key=lambda x: x.affinity, reverse=True)

    def remove_antibodies(self, antibodies, max_antibodies):
        antibodies = self.selection_fcn(antibodies)[:max_antibodies]
        return antibodies

    def point_mutation(self, clone, mutation_rate):
        """
        Iterates over each clone's paratope and mutates it accordingly to the mutation rate
        :param clone: The clone on which perform the mutation
        :param mutation_rate: The mutation's occurrence rate
        :return: The mutated clone
        """
        for i in range(0, len(clone.paratopes)):
            if random() < mutation_rate:
                clone.paratopes[i] = self.randomComponent()
        return clone

    def executive(self):

        config: AISConfig = self.config
        self._beginExecution()

        # Initialization of the variable contatining the index of the iteration and of the antibodies set
        iteration = 0
        antibodies = []
        memoryset = []
        best_antibody: AISAntibody = None

        # Antibodies creation
        for i in range(0, config.number_of_antibodies):
            antibodies.append(self.random_antibody_fcn())

        while self._isExecutable():

            # Increment the iteration number
            iteration += 1

            # Calculate affinity for each antibody
            self.calculate_affinity_fcn(antibodies)

            # Clonation
            clones = self.clone_antibodies_fcn(
                antibodies, config.clone_rate)

            # Hypermutation
            clones = self.mutation_fcn(clones, config.mutation_exp)

            # Computes the clones' affinity
            self.calculate_affinity_fcn(clones)

            # Add the clones to the antibodies list
            antibodies += clones

            # This is needed in order to remove identical/unnecessary antibodies
            antibodies = self.remove_antibodies(
                antibodies, config.max_antibodies)

            # Assignment of the best antibodies to the memory set
            memoryset = antibodies[:config.mem_size]

            if len(antibodies) - config.num_remove > 0:
                for i in range(len(antibodies) - config.num_remove, len(antibodies)):
                    antibodies[i] = self.random_antibody_fcn()

            fitness_improved = False
            if best_antibody is None or memoryset[0].affinity > best_antibody.affinity:
                best_antibody = memoryset[0].clone()
                fitness_improved = True

            # Print the current best solution and its value
            delta = 0
            if self.config.save_image_each == -1 and fitness_improved:
                delta = delta+self.saveImage(str(self.currentGen),
                                             best_antibody.paratopes)
            elif self.config.save_image_each >= 1 and self.currentGen % self.config.save_image_each == 0:
                delta = delta+self.saveImage(str(self.currentGen),
                                             best_antibody.paratopes)
            self._updateExecution(-best_antibody.affinity,
                                  best_antibody.paratopes,
                                  delta=delta)

        self._endExecution()
        return best_antibody.paratopes

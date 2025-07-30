from configparser import ConfigParser
import os
import random
import time
import json
import numpy as np
from imageHelper import ImageHelper
from statisticHelper import StatisticHelper

# all parameter values are bound between 0 and 1, later to be expanded:
BOUNDS_LOW, BOUNDS_HIGH = 0.0, 1.0  # boundaries for all dimensions


class AlgorithmConfigBase:
    def __init__(self, config: ConfigParser):
        self.polygon_size = 3
        self.number_of_polygon = 100
        self.save_image_each = 1000
        self.max_generation = 1000
        self.max_time = -1
        self.verbose = False
        self.objective_fun_method = "MSE"  # or SSIM
        self.target_solution = -1.0
        self.update(config)

    def toString(self, multiline=True):
        s = ""
        multi = " "
        if multiline:
            multi = "\n"
        l = self.__dict__.keys()
        for name in l:
            if name[0] != '_':
                value = self.__getattribute__(name)
                s = s+name+"="+str(value)+multi
        return s

    def update(self, config: ConfigParser):
        def getValue(name):
            try:
                return config['DEFAULT'][name]
            except:
                return None

        l = self.__dict__.keys()
        for name in l:
            if name[0] != '_':
                typev = type(self.__getattribute__(name))
                strv = getValue(name)
                if strv != None:
                    v = typev(strv)
                    self.__setattr__(name, v)


class AlgorithmBase:
    def __init__(self, config: AlgorithmConfigBase, image_file: str, output_folder: str, id: str):
        self.config = config
        self.image_file = image_file
        self.output_folder = output_folder
        self.id = id
        self.__statistic = None

        # create the image test class instance:
        self.image_helper = ImageHelper(
            self.image_file, config.polygon_size)

        # calculate total number of params in chromosome:
        # For each polygon we have:
        # two coordinates per vertex, 3 color values, one alpha value
        self.num_of_params = config.number_of_polygon * \
            (config.polygon_size * 2 + 4)

        # fitness calculation using MSE as difference metric:
        self.objectiveFunction = self.image_helper.getDifferenceFunc(
            self.config.objective_fun_method)

        # save inputs parameters
        input_file_path = os.path.join(self.output_folder, "inputs.txt")
        with open(input_file_path, "w") as input_file:
            input_file.write(self.config.toString(multiline=True))

    @property
    def currentGen(self):
        return self.__statistic.currentGen

    @property
    def isRunning(self):
        return self.__statistic is not None

    def randomSolution(self, low=BOUNDS_LOW, up=BOUNDS_HIGH):
        # helper function for creating random real numbers uniformly distributed within a given range [low, up]
        # it assumes that the range is the same for every dimension
        return [random.uniform(l, u) for l, u in zip([low] * self.num_of_params, [up] * self.num_of_params)]

    def randomComponent(self, low=BOUNDS_LOW, up=BOUNDS_HIGH):
        return random.uniform(low, up)

    def saveImage(self, name: str, polygonData: any, header=None):
        st = time.time()
        try:
            if not isinstance(polygonData, list):
                polygonData = list(polygonData)

            # create folder if does not exist:
            folder = os.path.join(self.output_folder, "results")
            if not os.path.exists(folder):
                os.makedirs(folder)
            imageCompareFilename = os.path.join(folder, name+"_compare.png")
            solutionFilename = os.path.join(folder, name+"_solution.txt")
            imageGeneratedFilename = os.path.join(
                folder, name+"_generated.bmp")

            # save the image in the folder:
            self.image_helper.saveImage(
                polygonData, imageCompareFilename, header)

            # save file data
            # txt = str(polygonData)
            # with open(solutionFilename, 'w') as f:
            #    f.write(txt)
            with open(solutionFilename, 'w') as f:
                json.dump(polygonData, f)

            # save image generated
            imageGenerated = self.image_helper.polygonDataToImage(polygonData)
            imageGenerated.save(imageGeneratedFilename, bitmap_format='bmp')

        except Exception as e:
            print("Error in saving image: ", e)

        et = time.time()
        return et-st

    def _beginExecution(self):
        output_file = os.path.join(self.output_folder, "statistic.txt")
        self.__statistic = StatisticHelper(output_file, self.config.verbose)
        pass

    def _updateExecution(self, fitness: float, current_solution,
                         fitness_worse: float = np.nan,
                         fitness_mean: float = np.nan,
                         fitness_std: float = np.nan,
                         image_save: bool = False,
                         delta: float = 0):
        self.__statistic.addRecord(fitness, fitness_worse,
                                   fitness_mean, fitness_std,  delta)

    def _endExecution(self):
        self.__statistic.close()
        self.__statistic = None

    def _isExecutable(self):
        if self.config.target_solution >= 0:
            return self.__statistic.currentFitness > self.config.target_solution
        elif self.config.max_time is not None and self.config.max_time >= 0:
            return self.__statistic.offtenTime <= self.config.max_time
        else:
            return self.__statistic.currentGen <= self.config.max_generation

    def executive(self):
        # abstract method
        pass

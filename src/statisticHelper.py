import time


class StatisticHelper:
    def __init__(self, filename: str, verbose=True) -> None:
        self.__filename = filename
        self.__file = open(filename, 'w')
        self.__current_gen = 0
        self.__current_fitness = None
        self.__start_time = time.time()
        self.__current_time = time.time()
        self.__sum_time = 0
        self.__enable_print = verbose

    @property
    def filename(self):
        return self.__filename

    @property
    def offtenTime(self):
        return self.__sum_time

    @property
    def currentGen(self):
        return self.__current_gen

    @property
    def currentFitness(self):
        return self.__current_fitness

    @property
    def enablePrint(self):
        return self.__enable_print

    @enablePrint.setter
    def set_enablePrint(self, value: bool):
        self.__enable_print = value

    @property
    def countRecords(self):
        return self.countRecords

    @property
    def isOpened(self):
        return self.__file != None

    def close(self):
        self.__file.close()
        self.__file = None

    def addRecord(self, fitness: float,  fitness_worse: float,
                  fitness_mean: float, fitness_std: float, delta=0):

        self.__current_fitness = fitness
        if self.__current_gen > 0:
            t = time.time()-self.__current_time-delta
        else:
            t = 0
            self.__file.write(
                "iteration\tfitness\ttime\ttotal_time\tfitness_worse\tfitness_mean\tfitness_std\n")
        self.__sum_time = self.__sum_time+t

        s = '\t'.join([str(self.__current_gen), str(fitness),
                      str(t), str(self.__sum_time), str(fitness_worse),
                      str(fitness_mean), str(fitness_std)])
        self.__file.write(s+"\n")
        self.__file.flush()

        if self.__enable_print:
            print(str(self.__current_gen)+") " + str(fitness) +
                  " "+str(t)+" " + str(self.__sum_time))

        self.__current_gen = self.__current_gen+1
        self.__current_time = time.time()

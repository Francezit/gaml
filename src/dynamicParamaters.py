
from collections import deque
import random
import time
from statistics import mean

DYNPRMS_METHOD_CODE_OLS = 1
DYNPRMS_PROBLEM_TYPE_MIN = -1
DYNPRMS_PROBLEM_TYPE_MAX = 1


class DynamicParameters:
    def __init__(self, buffer_size: int, setting: any, threshold=0.1, problem_type=DYNPRMS_PROBLEM_TYPE_MIN, method_code=DYNPRMS_METHOD_CODE_OLS, log_file=None):
        self.__buffer_size = buffer_size
        self.__parameter_len = len(setting)
        if problem_type != DYNPRMS_PROBLEM_TYPE_MIN and problem_type != DYNPRMS_PROBLEM_TYPE_MAX:
            raise Exception("problem_type not supported")
        self.__problem_type = problem_type
        self.__method_code = method_code
        if log_file != None:
            self.__log_file = open(log_file, 'w')
        else:
            self.__log_file = None

        self.__threshold = threshold
        self.__enable = True
        self.clear()

        self.__parameters = [0 for i in range(self.__parameter_len)]
        for i in range(self.__parameter_len):
            self.__parameters[i] = setting[i]['initial']

        self.__parameter_map = {}
        for i in range(self.__parameter_len):
            self.__parameter_map[setting[i]['name']] = i

        self.__parameters_factor = [setting[i]['factor']
                                    for i in range(self.__parameter_len)]
        self.__parameters_velocity = [
            (setting[i]['range'][1]-setting[i]['range'][0])/setting[i]['levels'] for i in range(self.__parameter_len)]
        self.__parameters_upperbound = [
            setting[i]['range'][1] for i in range(self.__parameter_len)]
        self.__parameters_lowerbound = [
            setting[i]['range'][0] for i in range(self.__parameter_len)]
        self.__parameters_enable = [
            setting[i]['enable'] for i in range(self.__parameter_len)]

    @property
    def enable(self):
        return self.__enable

    def setStatus(self, enable: bool):
        self.__enable = enable

    @property
    def capacity(self):
        return self.__buffer_size

    @property
    def length(self):
        if self.__buffer_enable:
            return self.__buffer_size
        else:
            return self.__buffer_pointer

    @property
    def empty(self):
        return self.__buffer_enable == False and self.__buffer_pointer == 0

    @property
    def problemType(self):
        return self.problemType

    @property
    def methodCode(self):
        return self.__method_code

    def sum(self):
        if self.empty == False:
            return self.__buffer_sum
        else:
            return None

    def mean(self):
        if self.empty == False:
            return self.__buffer_sum/self.length
        else:
            return None

    def values(self):
        if self.__buffer_enable:
            return list(self.__buffer)
        elif self.__buffer_pointer > 0:
            return self.__buffer[0:(self.__buffer_pointer-1)]
        else:
            return []

    def register(self, value: float):
        if not self.__enable:
            return

        if self.__buffer_pointer == self.__buffer_size-1:
            self.__buffer_enable = True

        old = self.__buffer[self.__buffer_pointer]
        self.__buffer[self.__buffer_pointer] = value
        self.__buffer_pointer = (self.__buffer_pointer+1) % self.__buffer_size
        self.__buffer_sum = self.__buffer_sum-old+value

        if self.__buffer_enable:
            trend, coeff = self.__predictor()
            self.__process(trend)
            self.__buffer_indexs.rotate(1)
        else:
            coeff = None

        if self.__log_file != None:
            st = time.time()
            gprms = self.getAll()
            s = "\t".join([str(x) for x in gprms])
            self.__log_file.write(str(value)+"\t"+s+"\t"+str(coeff)+'\n')
            self.__log_file.flush()
            et = time.time()
            return et-st
        else:
            return 0

    def get(self, parameter_name):
        return self.__parameters[self.__parameter_map[parameter_name]]

    def getParameterIndex(self, parameter_name):
        return self.__parameter_map[parameter_name]

    def getAt(self, parameter_index: int):
        return self.__parameters[parameter_index]

    def getParameterFunction(self, parameter_name):
        idx = self.getParameterIndex(parameter_name)
        enable = self.__parameters_enable[idx]
        if enable:
            def eval():
                return self.getAt(idx)
        else:
            r = self.getAt(idx)

            def eval():
                return r
        return eval

    def getAll(self):
        return self.__parameters

    def clear(self):
        self.__buffer_pointer = 0
        self.__buffer = [0.0 for i in range(self.__buffer_size)]
        self.__buffer_indexs = deque([i for i in range(self.__buffer_size)])
        self.__buffer_sum = 0
        self.__buffer_enable = False

    def dispose(self):
        self.clear()
        if self.__log_file != None:
            self.__log_file.close()
            self.__log_file = None

    def __predictor(self):
        X = list(self.__buffer_indexs)
        Y = self.__buffer

        if self.__method_code == DYNPRMS_METHOD_CODE_OLS:
            xm = mean(X)
            ym = mean(Y)
            n = sum([(X[i]-xm)*(Y[i]-ym) for i in range(self.__buffer_size)])
            d = sum([pow((X[i]-xm), 2) for i in range(self.__buffer_size)])
            p = n/d

            result = 0
            if p >= self.__threshold:
                result = 1
            elif p <= -self.__threshold:
                result = -1
            return result*self.__problem_type, p
        else:
            raise Exception("Method not supported")

    def __process(self, trend: int):
        for i in range(self.__parameter_len):
            current_value = self.__parameters[i]
            if trend <= -1:  # decrease
                self.__parameters[i] = self.__saturated(
                    current_value+self.__parameters_velocity[i], i)
            elif trend >= 1:  # increase
                self.__parameters[i] = self.__saturated(
                    current_value-self.__parameters_velocity[i], i)
            else:  # stuck
                self.__parameters[i] = self.__saturated(
                    current_value+(self.__parameters_velocity[i]/self.__parameters_factor[i]), i)

    def __saturated(self, value: float, parameter_index: int) -> float:
        l = self.__parameters_lowerbound[parameter_index]
        u = self.__parameters_upperbound[parameter_index]
        if value < l:
            return l
        elif value > u:
            return u
        else:
            return value


if __name__ == "__main__":
    # TEST
    setting = [{
        'name': 'p1',
        'initial': 0.1,
        'range': [0, 0.8],
        'factor':3,
        'levels':40
    }, {
        'name': 'p2',
        'initial': 0.4,
        'range': [0, 1],
        'factor':3,
        'levels':40
    },
        {
        'name': 'p3',
        'initial': 4,
        'range': [1, 10],
        'factor':3,
        'levels':40
    }
    ]
    register = DynamicParameters(10, setting)
    for i in range(50):
        register.register(random.gauss(0, 1))
        print(register.getAll())

    for i in range(50):
        register.register(10)
        print(register.getAll())

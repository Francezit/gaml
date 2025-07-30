from importlib import import_module
import sys
import getopt
import os
import random
from configparser import ConfigParser
from algorithmBase import AlgorithmBase, AlgorithmConfigBase


def main(argc, argv):
    opts, args = getopt.getopt(argv, "d:a:o:vi:t:n:c:s:", [
        "image_folder=",
        "algorithm=",
        "output_folder=",
        "verbose=",
        "config_file=",
        "irace_output=",
        "irace_id=",
        "custom_params=",
        "seed_value="
    ])

    IMAGE_FOLDER = "./images"
    ALGORITHM_NAME = None
    OUTPUT_FOLDER = None
    VERBOSE = False
    SEED_VALUE = None
    CUSTOM_PARMS = None
    CONFIG_FILE = None
    IRACE_OUTPUT = None
    IRACE_ID = "test"

    for opt, arg in opts:
        if opt in ("-d", "--image_folder"):
            IMAGE_FOLDER = arg
        elif opt in ("-a", "--algorithm"):
            ALGORITHM_NAME = arg
        elif opt in ("-o", "--output_folder"):
            OUTPUT_FOLDER = arg
        elif opt in ("-v", "--verbose"):
            VERBOSE = True
        elif opt in ("-i", "--config_file"):
            CONFIG_FILE = arg
        elif opt in ("-t", "--irace_output"):
            IRACE_OUTPUT = arg
        elif opt in ("-n", "--irace_id"):
            IRACE_ID = arg
        elif opt in ("-c", "--custom_params"):
            CUSTOM_PARMS = arg
        elif opt in ("-s", "--seed"):
            SEED_VALUE = int(arg)

    custom_parameters = []
    if CUSTOM_PARMS != None:
        for item in CUSTOM_PARMS.split(' '):
            custom_parameters.append(item)
        print(",".join(custom_parameters))

    if ALGORITHM_NAME == None:
        raise Exception("I don't know which algorithm I have to run!")

    if OUTPUT_FOLDER == None:
        raise Exception("I don't know where I have to save the results!")

    random.seed(SEED_VALUE)

    config = ConfigParser()
    if CONFIG_FILE != None and os.path.exists(CONFIG_FILE):
        config.read(CONFIG_FILE)

    if len(custom_parameters) > 0:
        for cmd in custom_parameters:
            s = cmd.split('=')
            config.set("DEFAULT", s[0], s[1])

    algorithm = ALGORITHM_NAME
    algorithm_module = import_module(algorithm)
    algorithm_class = getattr(algorithm_module, algorithm)
    algorithm_config_class = getattr(algorithm_module, algorithm+"Config")

    files = os.listdir(IMAGE_FOLDER)
    irace_eval_result = 0
    irace_eval_count = 0
    for file in files:
        try:
            image_name = os.path.splitext(os.path.basename(file))[0]
            image_file = os.path.join(IMAGE_FOLDER, file)

            print(OUTPUT_FOLDER+"::::"+image_name+"::::"+algorithm)
            output_folder = os.path.join(OUTPUT_FOLDER, image_name)
            if os.path.exists(output_folder) == False:
                os.makedirs(output_folder, exist_ok=True)

            print("Run-> "+image_name+"::"+algorithm+"::"+IRACE_ID)
            algorithm_config_instance: AlgorithmConfigBase = algorithm_config_class(
                config)
            if VERBOSE:
                algorithm_config_instance.verbose = VERBOSE
                print(algorithm_config_instance.toString(multiline=True))

            algorithm_instance: AlgorithmBase = algorithm_class(
                algorithm_config_instance,
                image_file,
                output_folder,
                IRACE_ID)
            result = algorithm_instance.executive()
            eval_result = algorithm_instance.objectiveFunction(result)
            algorithm_instance.saveImage(
                "final_result", result, str(eval_result))
            print(eval_result)
            irace_eval_result = irace_eval_result + eval_result
            irace_eval_count = irace_eval_count + 1
        except Exception as err:
            print(OUTPUT_FOLDER+"::::"+image_name +
                  "::::"+algorithm+"-ERROR-"+str(err))

    if IRACE_OUTPUT != None:
        with open(IRACE_OUTPUT, 'w') as f:
            f.write(str(int(irace_eval_result/irace_eval_count)))
    pass


if __name__ == "__main__":
    v = sys.argv[1:]
    main(len(v), v)

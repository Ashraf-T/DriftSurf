import logging
from drift_detection.__init__ import *
import training
import results
import read_data as data
import sys
import models

if __name__ == "__main__":

    if len(sys.argv) < 5:
        print("needs 4 arguments: dataset ({0}), computation (each_model, all_models), rho/lambda, base_learner".format(', '.join(data.read_dataset.AVAILABLE_DATASETS)))
        exit()

    dataset_name = sys.argv[1]
    computation = sys.argv[2]
    rate = int(sys.argv[3])
    base = sys.argv[4]

    results = results.Results(dataset_name)

    expt = training.Training(dataset=dataset_name, base_learner=base, computation=computation, rate=rate, algo_names=['Aware', 'MDDM', 'DriftSurf'])

    N = 1
    outputs = []
    for i in range(N):
        print({'Trial {0}'.format(i)})
        logging.info('Trial {0}'.format(i))
        output = expt.process(delta=0.1)
        outputs.append(output)

    results.gather_training_results(outputs,computation)

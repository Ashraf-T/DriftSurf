import logging
from drift_detection.__init__ import *
import training
import results
import read_data as data
import sys
import models

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("needs 3 arguments: dataset ({0}), computation (model, algorithm), rho/lambda".format(', '.join(data.read_dataset.AVAILABLE_DATASETS)))
        exit()

    dataset_name = sys.argv[1].lower()
    computation = sys.argv[2].lower()
    rate = int(sys.argv[3])
    opt = models.Opt.STRSAGA #'STRSAGA' #sys.argv[4].upper()

    results = results.Results(dataset_name)

    algo_names = ['Aware', 'MDDM', 'AUE', 'DriftSurf']
    expt = training.Training(dataset=dataset_name, computation=computation, rate=rate, base_learner=opt, algo_names=algo_names)

    N = 5
    outputs = []
    logging.info('Algorithms {0}, computation: {1}'.format(algo_names, computation))
    for i in range(N):
        print({'Trial {0}'.format(i)})
        logging.info('Trial {0}'.format(i))
        output = expt.process(delta=0.1, loss_fn='reg')
        outputs.append(output)

    results.gather_training_results(outputs,computation)

import logging
from drift_detection.__init__ import *
import training
import results
import read_data as data
import sys
import models

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("needs 3 arguments: dataset ({0}), computation (each_model, all_models), rho/lambda".format(', '.join(data.read_dataset.AVAILABLE_DATASETS)))
        exit()

    dataset_name = sys.argv[1]
    computation = sys.argv[2]
    rate = int(sys.argv[3])
    opt = models.Opt.STRSAGA.upper() #sys.argv[4].upper()

    results = results.Results(dataset_name)

    expt = training.Training(dataset=dataset_name, computation=computation, rate=rate, base_learner=opt, algo_names=['Aware', 'MDDM', 'AUE', 'DriftSurf', 'HAT', 'OBL'])
    #algo_names=['Aware', 'MDDM', 'AUE', 'Candor', 'DriftSurf']

    N = 1
    outputs = []
    for i in range(N):
        print({'Trial {0}'.format(i)})
        logging.info('Trial {0}'.format(i))
        output = expt.process(delta=0.1, loss_fn='reg')
        outputs.append(output)

    results.gather_training_results(outputs,computation)

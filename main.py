import logging
from drift_detection.__init__ import *
import training
import results
import read_data as data
import sys
import models
import numpy

if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("needs 3 arguments: dataset ({0}), computation (each_model, all_models), rho/lambda".format(', '.join(data.read_dataset.AVAILABLE_DATASETS)))
        exit()

    dataset_name = sys.argv[1]
    computation = sys.argv[2]
    rate = int(sys.argv[3])
    # opt = models.Opt.SGD.upper()
    opt = models.Opt.STRSAGA.upper() #sys.argv[4].upper()

    results = results.Results(dataset_name)

    # ['Aware', 'MDDM', 'AUE', 'DriftSurf', 'HAT']
    # HAT, OBL
    #algo_names=['Aware', 'MDDM', 'AUE', 'Candor', 'DriftSurf']
    algos = ['Standard']
    expt = training.Training(dataset=dataset_name, computation=computation, rate=rate, opt=opt, algo_names=algos, drift=True, base_learner='LR')

    N = 1
    outputs = []
    FPRs = []
    RTs = []
    for i in range(N):
        print({'Trial {0}'.format(i)})
        logging.info('Trial {0}'.format(i))
        # output = expt.process(delta=0.05, loss_fn='reg')
        output, FPR, RT = expt.process(delta=0.1, loss_fn='zero-one', drift_detectr=MDDM_G())
        outputs.append(output)
        FPRs.append(FPR)
        RTs.append(RT)

    false_positive_rate = {}
    avg_recovery_time = {}

    for key in FPRs[0].keys():
        false_positive_rate[key] = numpy.median([o[key] for o in FPRs])
        avg_recovery_time[key] = numpy.median([o[key] for o in RTs])

    print('avg_FPR: {0}, avg_RT: {1}'.format(false_positive_rate, avg_recovery_time))
    logging.info('avg_FPR: {0}, avg_RT: {1}'.format(false_positive_rate, avg_recovery_time))
    results.gather_training_results(outputs, computation, algorithms=algos)

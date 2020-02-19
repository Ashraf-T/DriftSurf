import logging
from drift_detection.__init__ import *
import Training
import Results
import Sensitivity_r
import Sensitivity_noise



if __name__ == "__main__":

    dataset_name = 'sea'
    results = Results.Results(dataset_name)

    # -------------- Training ------------------
    computation = 'unlimited'
    rate = 2
    drift_detector = MDDM_G()

    expt = Training.Training(dataset_name, computation, rate)

    N = 5
    outputs = []
    for i in range(N):
        print({'Trial {0}'.format(i)})
        logging.info('Trial {0}'.format(i))
        output = expt.process(drift_detectr=drift_detector, delta=0.1, loss_fn='reg')
        outputs.append(output)

    results.gather_training_results(outputs,computation)

    # -------------- Sensitivity_r ------------------
    # list_r = [2,4,6,8,10]
    # sens = Sensitivity_r.Sensitivity_r(dataset_name, list_r)
    # outputs = sens.process(loss_fn='zero-one')
    # results.plot_sensitivity_r(outputs,list_r)

    # -------------- Sensitivity_noise ---------------
    # sens_n = Sensitivity_noise.Sensitivity_noise()
    # outputs = sens_n.process()
    # results.plot_sensitivity_noise(outputs)

import models
import hyperparameters
import random
import Training
from drift_detection.__init__ import *
import Results

class Sensitivity_noise:

    OPT = models.Opt.STRSAGA
    def __init__(self):

        self.dataset_name = 'sea'
        self.noise_levels = [0, 10, 20, 30]
        self.algo_names=['MDDM','AUE','DSURF']
        self.ave_over_time = {}

        for algo in self.algo_names:
            self.ave_over_time[algo] = []

    def process(self, delta=0.1, loss_fn='zero_one', drift_detector=MDDM_G()):

        for noise in self.noise_levels:
            dataset_name = 'sea{0}'.format(noise)
            train = Training.Training(dataset_name, algo_names=self.algo_names)
            output = [train.process(delta,loss_fn,drift_detector)]

            ave, _ = Results.Results.average_over_time(output)
            for algo in ave.keys():
                self.ave_over_time[algo].append(ave[algo])

        return self.ave_over_time
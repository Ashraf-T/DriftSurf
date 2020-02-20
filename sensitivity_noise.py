import models
import hyperparameters
import random
import training
from drift_detection.__init__ import *
import results

class Sensitivity_noise:

    OPT = models.Opt.STRSAGA
    def __init__(self):

        self.dataset_name = 'sea'
        self.noise_levels = [0, 10, 20, 30]
        self.algo_names=['MDDM','AUE','DriftSurf']
        self.ave_over_time = {}

        for algo in self.algo_names:
            self.ave_over_time[algo] = []

    def process(self, delta=0.1, loss_fn='zero_one', drift_detector=MDDM_G()):

        for noise in self.noise_levels:
            dataset_name = 'sea{0}'.format(noise)
            print('Processing {0}'.format(dataset_name))
            train = training.Training(dataset_name, algo_names=self.algo_names)
            output = [train.process(delta,loss_fn,drift_detector)]

            ave = results.average_over_time(output)
            for algo in ave.keys():
                self.ave_over_time[algo].append(ave[algo])

        return self.ave_over_time

if __name__ == "__main__":

    results = results.Results('sea')
    sens_n = Sensitivity_noise()
    outputs = sens_n.process()
    results.plot_sensitivity_noise(outputs)

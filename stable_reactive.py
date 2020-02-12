import numpy
import random
import time
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import models as models
import math
from matplotlib.patches import Polygon
# import read_data as data
# import syn_data as syn_data
import dataset as data
import logging
from drift_detection.__init__ import *

# dataset = {
#     'rcv': {'n': 20242, 'd': 47237, 'b': 100, 'step_size': 5e-1, 'mu': 1e-5, 'T_reactive': 4,
#             'drift_time': [30, 60]},
#     'covtype': {'n': 581012, 'd': 55, 'b': 100, 'step_size': 5e-3, 'mu': 1e-4, 'T_reactive': 4,
#                 'drift_time': [30, 60]},
#     'pow': {'n': 29928, 'd': 3, 'b': 100, 'step_size': 2e-2, 'mu': 1e-3, 'T_reactive': 4,
#             'drift_time': [17, 47, 76]},
#     'air-100': {'n': 58100, 'd': 13, 'b': 100, 'step_size': 2e-2, 'mu': 1e-3, 'T_reactive': 4,
#                 'drift_time': [31, 67]},
#     'elec': {'n': 45312, 'd': 14, 'b': 34, 'step_size': 1e-1, 'mu': 1e-4, 'T_reactive': 1, 'drift_time': [20]},
#     'sea': {'n': 100000, 'd': 4, 'b': 100, 'step_size': 1e-3, 'mu': 1e-2, 'T_reactive': 4,
#             'drift_time': [25, 50, 75]},
#     'hyperplane_slow': {'n': 100000, 'd': 3, 'b': 100, 'step_size': 1e-1, 'mu': 1e-3, 'T_reactive': 4,
#                         'drift_time': []},
#     'hyperplane_fast': {'n': 100000, 'd': 11, 'b': 100, 'step_size': 1e-2, 'mu': 1e-3, 'T_reactive': 4,
#                         'drift_time': []},
#     'sine1_new': {'n': 10000, 'd': 3, 'b': 100, 'step_size': 2e-1, 'mu': 1e-3, 'T_reactive': 4,
#                   'drift_time': [20, 40, 60, 80]}
# }

class experiments():

    OPT = models.Opt.SAGA
    OPT_sgd = models.Opt.SGD

    def __init__(self, dataset:str, r=4, delta=0.2, rate=2, loss_fn='zero-one', algo_names=['Aware','DD', 'AUE', 'DSURF']):

        self.algorithms = algo_names
        self.loss = {}

        self.dataset_name = dataset
        self.X, self.Y, self.n, self.d, self.mu, self.step_size, self.b, self.drift_times = getattr(data, dataset_name)()
        self.lam = int(self.n//self.b)
        self.rho = int(self.lam * rate)
        self.S = 0

        self.DSURF_delta = delta
        self.DSURF_loss_fn = loss_fn
        self.DSURF_t = 0
        self.DSURF_r = r

        self.DSURF = models.LogisticRegression_DSURF(self.d, experiments.OPT, self.DSURF_delta, self.DSURF_loss_fn)
        self.AUE = models.LogisticRegression_AUE(self.d, experiments.OPT)
        self.DD = models.LogisticRegression_expert(numpy.random.rand(self.d), experiments.OPT)
        self.Aware = models.LogisticRegression_expert(numpy.random.rand(self.d), experiments.OPT)

        self.DD_drift_detector = MDDM_G()

        for algo in self.algorithms:
            self.loss[algo] = [0] * self.b

        logging.info('initialized')
        logging.info('1 - dataset : {0}, n : {1}, b : {2}'.format(self.dataset_name, self.n, self.b))


    def read_data(dataset_name):
        return getattr(data, dataset_name)()

    def update_loss(self, test_set, time):

        for algo in self.algorithms:
            self.loss[algo][time] = getattr(self, algo).zero_one_loss(test_set)


    def process_DD(self, time, new_batch):

        if (self.DD_drift_detector.test(self.DD, new_batch) and time != 0):
            self.DD = models.LogisticRegression_expert(numpy.random.rand(self.d), self.OPT, self.S)
            self.DD_drift_detector.reset()
            logging.info('DD drift detected, reset model : {0}'.format(time))

        self.update_model(self.DD)

    def process_AUE(self, time, new_batch):
        self.AUE.update_weights(new_batch)
        logging.info('AUE Experts at time {0}: {1}'.format(time, [int(k / self.lam) for k in self.AUE.weights.keys()]))
        for index, expert in self.AUE.experts.items():
            self.update_model(expert)

    def update_model(self, model):
        if model:
            lst = list(model.T_pointers)
            for s in range(self.rho):
                if s % 2 == 0 and lst[1] < self.S + self.lam:
                    j = lst[1]
                    lst[1] += 1
                else:
                    j = random.randrange(lst[0], lst[1])
                point = (j, self.X[j], self.Y[j])
                model.update_step(point, self.step_size, self.mu)
            model.update_effective_set(lst[1])

    def process_DSURF(self, time, new_batch):
        self.DSURF.update_perf_all(new_batch, self.mu)

        if self.DSURF.stable:
            if self.DSURF.enter_reactive(self.S, new_batch, self.mu):
                self.DSURF_t = 0
                logging.info('DSURF enters reactive state : {0}'.format(time))

            else:
                # update models
                self.update_model(self.DSURF.expert_predictive)
                self.update_model(self.DSURF.expert_stable)

        if not self.DSURF.stable:
            # update models
            self.DSURF.update_reactive_sample_set(new_batch)
            self.update_model(self.DSURF.expert_predictive)
            self.update_model(self.DSURF.expert_reactive)

            self.DSURF_t += 1
            if self.DSURF_t == self.DSURF_r :
                self.DSURF.exit_reactive(self.S+self.lam, self.mu)

    def process(self):

        for time in range(self.b):

            print(time)

            if time in self.drift_times:
                self.Aware = models.LogisticRegression_expert(numpy.random.rand(self.d), self.OPT, self.S)

            # measure accuracy over upcoming batch
            test_set = [(i, self.X[i], self.Y[i]) for i in range(self.S, self.S + self.lam)]
            self.update_loss(test_set, time)

            # AUE
            self.process_AUE(time, test_set)

            # DD
            self.process_DD(time, test_set)

            # DSURF
            self.process_DSURF(time, test_set)

            # Aware
            self.update_model(self.Aware)

            self.S += self.lam

        return self.loss


def plot(output:dict, dataset_name, b, path, rate):

    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['lines.markersize'] = 4

    b = min(b, 100) # if b > 100, plot results of 100 time steps per plot

    for i in range(max(math.ceil(b / 100), 1)):

        t = 1
        first = max(1, i * b)
        last = min(b * (i + 1), b)

        xx = range(first, last, t)

        # ------------ accuracy  --------------
        plt.figure(1)
        plt.clf()
        colors = ['black', 'green', 'red', 'blue']
        markers = ['^', 's', 'o', 'x']
        k = 0
        for key in output.keys():
            plt.plot(xx, output[key][first:last], colors[k], label=key, marker = markers[k], linestyle='dashed',
                 markevery=10)
            k += 1
        plt.xlabel('Time')
        plt.ylabel('Misclassification rate')
        plt.legend()
        plt.xlim(first, last)
        plt.savefig(os.path.join(path, '{0}r{1}-acc{2}.eps'.format(dataset_name, rate, i)), format='eps')
        plt.savefig(os.path.join(path, '{0}r{1}-acc{2}.png'.format(dataset_name, rate, i)), format='png', dpi=200)


def median_outputs(output_list):
    output = {}
    if len(output_list) == 0:
        print('Error: no result')
    else:
        for key in output_list[0].keys():
            b = len(output_list[0][key])
            output[key] = [0] * b

    for t in range(b):
        for key in output.keys():
            output[key][t] = numpy.median([o[key][t] for o in output_list])

    return output, b

def average_over_time(output_list):

    if len(output_list) == 0:
        print('Error: no output')
    else:
        for key in output_list[0].keys():
            b = len(output_list[0][key])
            ave = 0
            for t in range(b):
                ave += sum([o[key][t] for o in output_list])
            ave /= (b * len(output_list))

            logging.info('average over time {0} : {1}'.format(key, ave))

def create_folder():
    current_time = time.strftime('%Y-%m-%d_%H%M%S')

    path = os.path.join('output', current_time)
    os.makedirs(path)

    return path

if __name__ == "__main__":

    dataset_name = 'rcv'
    path = create_folder()
    logging.basicConfig(filename=os.path.join(path,'{0}.log'.format(dataset_name)), filemode='w', level=logging.INFO)

    N = 5
    outputs = []
    for i in range(N):
        print ({'Trial {0}'.format(i)})
        logging.info('Trial {0}'.format(i))
        expt = experiments(dataset_name, 4)
        output = expt.process()
        outputs.append(output)

    average_over_time(outputs)

    with open(os.path.join(path,'data.pkl'), 'wb') as f:
        pickle.dump(outputs, f)

    with open(os.path.join(path,'data.pkl'), 'rb') as f:
        outputs = pickle.load(f)

    output, b = median_outputs(outputs)

    plot(output, dataset_name, b, path, 2)

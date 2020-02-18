import numpy
import random
import time
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import pickle
import models as models
import math
import read_data as data
import logging
from drift_detection.__init__ import *

MU = {'rcv': 1e-5, 'covtype': 1e-4, 'powersupply': 1e-3, 'airline': 1e-3, 'elec': 1e-4, 'sea': 1e-2, 'sine1': 1e-3, 'hyperplane_slow': 1e-3, 'hyperplane_fast': 1e-3 }
STEP_SIZE = {'rcv': 5e-1, 'covtype': 5e-3, 'powersupply': 2e-2, 'airline': 2e-2, 'elec': 1e-1, 'sea': 1e-3, 'sine1': 2e-1, 'hyperplane_slow' : 1e-1, 'hyperplane_fast': 1e-2}
b = {'default': 100, 'elec':34}
r = {'default': 4, 'elec': 1}

class Training():

    OPT = models.Opt.SAGA
    OPT_sgd = models.Opt.SGD
    LIMITED = 'limited'

    def __init__(self, dataset:str, computation='unlimited',algo_names=['Aware', 'SGD', 'OBL', 'MDDM', 'AUE', 'DSURF']):
        """

        :param dataset:
        :param algo_names:
        """

        self.algorithms = algo_names
        self.loss = {}
        self.computation =  computation
        # self.rate = 4 if self.computation == Training.LIMITED else 2
        self.rate = 2

        self.dataset_name = dataset
        readData = data.read_dataset()
        self.X, self.Y, self.n, self.d, self.drift_times = readData.read(self.dataset_name)

        if self.dataset_name.startswith('sea'):
            self.dataset_name = 'sea'

        self.mu = MU[self.dataset_name]
        self.step_size = STEP_SIZE[self.dataset_name]
        self.b = b[self.dataset_name] if self.dataset_name in b.keys() else b['default']
        self.lam = int(self.n//self.b)
        self.rho = self.lam * self.rate

    def update_loss(self, test_set, time):
        """

        :param test_set:
        :param time:
        :return:
        """

        for algo in self.algorithms:
            self.loss[algo][time] = getattr(self, algo).zero_one_loss(test_set)

    def setup_algorithms(self, delta, loss_fn, detector=MDDM_G()):
        """

        :param delta:
        :param loss_fn:
        :param detector:
        :return:
        """

        for algo in self.algorithms:
            if algo == 'DSURF': self.setup_DSURF(delta, loss_fn)
            elif algo == 'MDDM': self.setup_MDDM(detector)
            else: getattr(self, 'setup_{0}'.format(algo))()

    def setup_DSURF(self, delta, loss_fn):
        """

        :param delta:
        :param loss_fn:
        :return:
        """

        self.DSURF_t = 0
        self.DSURF_r = r[dataset_name] if self.dataset_name in r.keys() else r['default']
        self.DSURF = models.LogisticRegression_DSURF(self.d, Training.OPT, delta, loss_fn)

    def setup_MDDM(self, detector=MDDM_G()):
        """

        :param detector:
        :return:
        """
        self.MDDM = models.LogisticRegression_expert(numpy.random.rand(self.d), Training.OPT)
        self.MDDM_drift_detector = detector
        print('here', self.MDDM_drift_detector.__str__())

    def setup_AUE(self):
        """

        :return:
        """
        self.AUE = models.LogisticRegression_AUE(self.d, Training.OPT)

    def setup_Aware(self):
        """

        :return:
        """
        self.Aware = models.LogisticRegression_expert(numpy.random.rand(self.d), Training.OPT, self.S)

    def setup_SGD(self):
        """

        :return:
        """
        self.SGD = models.LogisticRegression_expert(numpy.random.rand(self.d), Training.OPT_sgd)

    def setup_OBL(self):
        """

        :return:
        """
        self.OBL = models.LogisticRegression_expert(numpy.random.rand(self.d), Training.OPT)

    def update_STRSAGA_model(self, model):
        """

        :param model:
        :return:
        """
        if model:
            weight = model.get_weight() if self.computation == Training.LIMITED else 1
            lst = list(model.T_pointers)
            for s in range(int(self.rho * weight)):
                if s % 2 == 0 and lst[1] < self.S + self.lam:
                    j = lst[1]
                    lst[1] += 1
                else:
                    j = random.randrange(lst[0], lst[1])
                point = (j, self.X[j], self.Y[j])
                model.update_step(point, self.step_size, self.mu)
            model.update_effective_set(lst[1])

    def process_MDDM(self, time, new_batch):
        """

        :param time:
        :param new_batch:
        :return:
        """

        if (self.MDDM_drift_detector.test(self.MDDM, new_batch) and time != 0):
            self.MDDM = models.LogisticRegression_expert(numpy.random.rand(self.d), self.OPT, self.S)
            self.MDDM_drift_detector.reset()
            logging.info('MDDM drift detected, reset model : {0}'.format(time))

        self.update_STRSAGA_model(self.MDDM)

    def process_AUE(self, time, new_batch):
        """

        :param time:
        :param new_batch:
        :return:
        """
        self.AUE.update_weights(new_batch)
        # logging.info('AUE Experts at time {0}: {1}'.format(time, [int(k / self.lam) for k in self.AUE.weights.keys()]))
        logging.info('AUE Experts at time {0}: {1}'.format(time, [int(k / self.lam) for k in self.AUE.experts.keys()]))
        for index, expert in self.AUE.experts.items():
            self.update_STRSAGA_model(expert)

    def process_DSURF(self, time, new_batch):
        """

        :param time:
        :param new_batch:
        :return:
        """
        self.DSURF.update_perf_all(new_batch, self.mu)

        if self.DSURF.stable:
            if self.DSURF.enter_reactive(self.S, new_batch, self.mu):
                self.DSURF_t = 0
                logging.info('DSURF enters reactive state : {0}'.format(time))

            else:
                # update models
                self.update_STRSAGA_model(self.DSURF.expert_predictive)
                self.update_STRSAGA_model(self.DSURF.expert_stable)

        if not self.DSURF.stable:
            # update models
            self.DSURF.update_reactive_sample_set(new_batch)
            self.update_STRSAGA_model(self.DSURF.expert_predictive)
            self.update_STRSAGA_model(self.DSURF.expert_reactive)

            self.DSURF_t += 1
            if self.DSURF_t == self.DSURF_r :
                self.DSURF.exit_reactive(self.S+self.lam, self.mu)

    def process_Aware(self):
        """

        :return:
        """
        self.update_STRSAGA_model(self.Aware)

    def process_OBL(self, total = 4):
        """

        :return:
        """
        lst = list(self.OBL.T_pointers)
        for s in range(self.rho):
            if s % 2 == 0 and lst[1] < self.S + self.lam:
                lst[1] += 1
                j = random.randrange(lst[0], lst[1])
            point = (j, self.X[j], self.Y[j])
            self.OBL.update_step(point, self.step_size, self.mu)
        self.OBL.update_effective_set(lst[1])

    def process_SGD(self):
        """

        :return:
        """
        sgdOnline_T = self.S
        for s in range(min(self.lam, self.rho)):
            if sgdOnline_T < self.S + self.lam:
                j = sgdOnline_T
                sgdOnline_T += 1
            point = (j, self.X[j], self.Y[j])
            self.SGD.update_step(point, self.step_size, self.mu)

    def process(self, delta=0.2, loss_fn='zero-one', drift_detectr=MDDM_G()):
        """

        :param rate:
        :param delta:
        :param loss_fn:
        :return:
        """

        self.S = 0
        # self.rho = int(self.lam * rate)
        self.setup_algorithms(delta, loss_fn, drift_detectr)

        for algo in self.algorithms:
            self.loss[algo] = [0] * self.b

        logging.info('dataset : {0}, n : {1}, b : {2}'.format(self.dataset_name, self.n, self.b))


        for time in range(self.b):

            print(time)

            # measure accuracy over upcoming batch
            test_set = [(i, self.X[i], self.Y[i]) for i in range(self.S, self.S + self.lam)]
            self.update_loss(test_set, time)

            if time in self.drift_times:
                self.setup_Aware()

            for algo in self.algorithms:
                if algo in ['SGD', 'OBL', 'Aware']: getattr(self, 'process_{0}'.format(algo))()
                else: getattr(self, 'process_{0}'.format(algo))(time, test_set)


            self.S += self.lam

        return self.loss


class Results:

    def __init__(self, dataset_name):
        """

        :param dataset_name:
        """
        self.dataset_name = dataset_name
        self.path = self.create_folder()
        logging.basicConfig(filename=os.path.join(self.path,'{0}.log'.format(dataset_name)), filemode='w', level=logging.INFO)

    def gather(self, output_list, computation):
        """

        :param output_list:
        :param rate:
        :return:
        """

        self.store_results(output_list)

        self.average_over_time(output_list)
        output, b = self.median_outputs(output_list)
        self.plot(output, self.dataset_name, b, self.path, computation)
        logging.shutdown()

    def store_results(self, output_list):
        """

        :param output_list:
        :return:
        """
        with open(os.path.join(self.path, 'data.pkl'), 'wb') as f:
            pickle.dump(output_list, f)

    def load_results(self):
        """

        :return:
        """
        with open(os.path.join(self.path, 'data.pkl'), 'rb') as f:
            outputs = pickle.load(f)
        return outputs

    def plot(self, output:dict, dataset_name, b, path, computation, algorithms=['Aware', 'MDDM', 'AUE', 'DSURF']):
        """

        :param output:
        :param dataset_name:
        :param b:
        :param path:
        :param rate:
        :return:
        """

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
            colors = ['black', 'green', 'red', 'blue', 'brown', 'magenta']
            markers = ['^', 's', 'o', 'x', '.', '+']
            k = 0
            for key in algorithms:
                linestyle = '-' if key == 'Aware' else '--'
                plt.plot(xx, output[key][first:last], colors[k], label=key, marker = markers[k], linestyle=linestyle,
                     markevery=10)
                k += 1
            plt.xlabel('Time')
            plt.ylabel('Misclassification rate')
            plt.legend()
            plt.xlim(first, last)
            plt.savefig(os.path.join(path, '{0}-{1}-acc{2}.eps'.format(dataset_name, computation, i)), format='eps')
            plt.savefig(os.path.join(path, '{0}-{1}-acc{2}.png'.format(dataset_name, computation, i)), format='png', dpi=200)

    def median_outputs(self, output_list):
        """

        :param output_list:
        :return:
        """
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

    def average_over_time(self, output_list):
        """

        :param output_list:
        :return:
        """
        output = {}

        if len(output_list) == 0:
            print('Error: no result')
        else:
            for key in output_list[0].keys():
                b = len(output_list[0][key])
                output[key] = [0] * b

                for t in range(b):
                    output[key][t] = numpy.mean([o[key][t] for o in output_list])
                print('average over time {0} : {1}'.format(key, numpy.mean(output[key])))
                print('variance over time {0} : {1}'.format(key, numpy.var(output[key])))
                logging.info('average over time {0} : {1}'.format(key, numpy.mean(output[key])))
                logging.info('variance over time {0} : {1}'.format(key, numpy.var(output[key])))

    def create_folder(self):
        """

        :return:
        """
        current_time = time.strftime('%Y-%m-%d_%H%M%S')

        path = os.path.join('output', current_time)
        os.makedirs(path)

        return path

if __name__ == "__main__":

    dataset_name = 'hyperplane_slow'
    computation = 'unlimited'
    drift_detector = MDDM_A()
    
    expt = Training(dataset_name, computation)
    results = Results(dataset_name)

    N = 5
    outputs = []
    for i in range(N):
        print({'Trial {0}'.format(i)})
        logging.info('Trial {0}'.format(i))
        output = expt.process(drift_detectr=drift_detector, delta=0.1, loss_fn='reg')
        outputs.append(output)

    results.gather(outputs,computation)

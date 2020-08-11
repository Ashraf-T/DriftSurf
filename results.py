import logging
import os
import pickle
import matplotlib as mpl
import math
import matplotlib.pyplot as plt
import numpy
import time
import hyperparameters
import read_data as data
import models

class Results:

    def __init__(self, dataset_name):

        self.dataset_name = dataset_name
        self.b = hyperparameters.b[self.dataset_name] if self.dataset_name in hyperparameters.b.keys() else hyperparameters.b['default']
        self.path = self.create_folder()
        logging.basicConfig(filename=os.path.join(self.path, '{0}.log'.format(dataset_name)), filemode='w', level=logging.INFO)

    def gather_training_results(self, output_list, computation):

        self.store_results(output_list, self.path)

        ave_over_time = self.average_over_time(output_list)
        output, b = self.median_outputs(output_list)
        self.store_results(ave_over_time, self.path, 'ave_over_time')
        self.plot_training(output, self.dataset_name, b, self.path, computation)
        logging.shutdown()

    @staticmethod
    def plot_sensitivity_r(dataset_name, output, list_r, path):

        if len(output) == 0:
            print('no output')
            exit()

        mpl.rcParams['lines.linewidth'] = 1.0
        mpl.rcParams['lines.markersize'] = 4

        b = hyperparameters.b[dataset_name] if dataset_name in hyperparameters.b.keys() else hyperparameters.b['default'] # if b > 100, plot results of 100 time steps per plot

        t = 1
        xx = range(1, b, t)

        # ------------ accuracy  --------------
        plt.figure(1)
        plt.clf()
        colors = ['black', 'green', 'red', 'blue', 'brown', 'magenta']
        markers = ['^', 's', 'o', 'x', '.', '+']
        k = 0
        for r in list_r:
            linestyle = '-' if r == 4 else '--'
            plt.plot(xx, output[r][1:b], colors[k], label='r={0}'.format(r), marker=markers[k],
                     linestyle=linestyle, markevery=10)
            k += 1
        plt.xlabel('Time')
        plt.ylabel('Misclassification rate')
        plt.legend()
        plt.xlim(1, b)
        plt.savefig(os.path.join(path, '{0}-r_sensitivity.eps'.format(dataset_name)), format='eps')
        plt.savefig(os.path.join(path, '{0}-r_sensitivity.png'.format(dataset_name)), format='png', dpi=200)

    @staticmethod
    def plot_greedy(dataset_name, output, path):

        if len(output) == 0:
            print('no output')
            exit()

        mpl.rcParams['lines.linewidth'] = 1.0
        mpl.rcParams['lines.markersize'] = 4

        b = hyperparameters.b[dataset_name] if dataset_name in hyperparameters.b.keys() else hyperparameters.b['default'] # if b > 100, plot results of 100 time steps per plot

        t = 1
        xx = range(1, b, t)

        # ------------ accuracy  --------------
        plt.figure(1)
        plt.clf()
        markers = ['o', 'x']
        k = 0
        for method in [models.LogisticRegression_DriftSurf.GREEDY, 'no-Greedy']:
            linestyle = '-' if method == models.LogisticRegression_DriftSurf.GREEDY else '--'
            label = 'DriftSurf' if method == models.LogisticRegression_DriftSurf.GREEDY else 'DriftSurf(no-greedy)'
            plt.plot(xx, output[method][1:b], 'k', label=label, marker=markers[k],
                     linestyle=linestyle, markevery=10)
            k += 1
        plt.xlabel('Time')
        plt.ylabel('Misclassification rate')
        plt.legend()
        plt.xlim(1, b)
        plt.savefig(os.path.join(path, '{0}-greedy.eps'.format(dataset_name)), format='eps')
        plt.savefig(os.path.join(path, '{0}-greedy.png'.format(dataset_name)), format='png', dpi=200)

    @staticmethod
    def plot_sensitivity_r_all(outputs, list_r, path):

        mpl.rcParams['lines.linewidth'] = 1.0
        mpl.rcParams['lines.markersize'] = 4

        xx = list_r

        # ------------ accuracy  --------------
        plt.figure(1)
        plt.clf()
        colors = ['black', 'green', 'red', 'blue', 'brown', 'magenta', 'teal', 'tomato', 'lime', 'tan', 'cyan', 'yellow']
        markers = ['^', 's', 'o', 'x', '.', '+', ',', 'v', '<', '>', 'd', '*']
        k = 0
        for dataset_name in data.read_dataset.AVAILABLE_DATASETS:
            plt.plot(xx, outputs[dataset_name], colors[k], label=dataset_name, marker=markers[k])
            k += 1
        plt.xlabel('r')
        plt.ylabel('Average of Misclassification over time')
        # plt.legend()
        # Put a legend to the right of the current axis
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.savefig(os.path.join(path, 'r_sensitivity-all.eps'), format='eps', bbox_inches='tight')
        plt.savefig(os.path.join(path, 'r_sensitivity-all.png'), format='png', dpi=200, bbox_inches='tight')

    def plot_sensitivity_noise(self, output):

        mpl.rcParams['lines.linewidth'] = 1.0
        mpl.rcParams['lines.markersize'] = 4

        xx = [0, 10, 20, 30]
        plt.figure(1)
        plt.clf()
        colors = ['green', 'red', 'blue']
        markers = ['s', 'o', 'x']
        k = 0
        for key in output.keys():
            plt.plot(xx, output[key], colors[k], label=key, marker=markers[k])
            k += 1
        plt.xlabel('noise_level')
        plt.ylabel('Misclassification rate')
        plt.legend()
        # plt.xlim(first, last)
        plt.savefig(os.path.join(self.path, '{0}-noise_sensitivity.eps'.format(self.dataset_name)), format='eps')
        plt.savefig(os.path.join(self.path, '{0}-noise_sensitivity.png'.format(self.dataset_name)), format='png', dpi=200)

    @staticmethod
    def store_results(output_list, path, name='data'):

        with open(os.path.join(path, '{0}.pkl'.format(name)), 'wb') as f:
            pickle.dump(output_list, f)

    @staticmethod
    def load_results(path):

        with open(os.path.join(path, 'data.pkl'), 'rb') as f:
            outputs = pickle.load(f)
        return outputs

    @staticmethod
    def plot_training(output:dict, dataset_name, b_in, path, computation, algorithms=['AUE']):
        # algorithms=['Aware', 'MDDM', 'AUE', 'DriftSurf']

        mpl.rcParams['lines.linewidth'] = 1.0
        mpl.rcParams['lines.markersize'] = 4

        b = min(b_in, 100)  # if b > 100, plot results of 100 time steps per plot

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
                label = 'DriftSurf' if key == 'DSURF' else key
                label = '1PASS-SGD' if key == 'SGD' else label
                plt.plot(xx, output[key][first:last], colors[k], label=label, marker=markers[k], linestyle=linestyle,
                     markevery=10)
                k += 1
            plt.xlabel('Time')
            plt.ylabel('Misclassification rate')
            plt.legend()
            plt.xlim(first, last)
            plt.savefig(os.path.join(path, '{0}-{1}-acc{2}.eps'.format(dataset_name, computation, i)), format='eps')
            plt.savefig(os.path.join(path, '{0}-{1}-acc{2}.png'.format(dataset_name, computation, i)), format='png', dpi=200)

    @staticmethod
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

    @staticmethod
    def average_over_time(output_list):

        output = {}

        if len(output_list) == 0:
            print('Error: no result')
        else:
            ave_over_time = {}
            # var_over_time = {}
            for key in output_list[0].keys():
                b = len(output_list[0][key])
                output[key] = [0] * b

                for t in range(b):
                    output[key][t] = numpy.mean([o[key][t] for o in output_list])

                ave_over_time[key] = numpy.mean(output[key])
                # var_over_time[key] = numpy.var(output[key])
                print('average over time {0} : {1}'.format(key, ave_over_time[key]))
                # print('variance over time {0} : {1}'.format(key, var_over_time[key]))
                logging.info('average over time {0} : {1}'.format(key, ave_over_time[key]))
                # logging.info('variance over time {0} : {1}'.format(key, var_over_time[key]))
        return ave_over_time

    @staticmethod
    def create_folder():

        current_time = time.strftime('%Y-%m-%d_%H%M%S')

        path = os.path.join('output', current_time)
        os.makedirs(path)

        return path

import logging
import os
import pickle
import matplotlib as mpl
import math
import matplotlib.pyplot as plt
import numpy
import time
import hyperparameters


class Results:

    DSURF = 'DSURF'

    def __init__(self, dataset_name):
        """

        :param dataset_name:
        """
        self.dataset_name = dataset_name
        self.b = hyperparameters.b[self.dataset_name] if self.dataset_name in hyperparameters.b.keys() else hyperparameters.b['default']
        self.path = self.create_folder()
        logging.basicConfig(filename=os.path.join(self.path, '{0}.log'.format(dataset_name)), filemode='w', level=logging.INFO)

    def gather_training_results(self, output_list, computation):
        """

        :param output_list:
        :param computation:
        :return:
        """

        self.store_results(output_list, self.path)

        self.average_over_time(output_list)
        output, b = self.median_outputs(output_list)
        self.plot_training(output, self.dataset_name, b, self.path, computation)
        logging.shutdown()

    def plot_sensitivity_r(self, output, list_r):
        """

        :param output:
        :param r_max:
        :param r_min:
        :return:
        """

        mpl.rcParams['lines.linewidth'] = 1.0
        mpl.rcParams['lines.markersize'] = 4

        b = min(self.b, hyperparameters.b['default'])  # if b > 100, plot results of 100 time steps per plot

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
            for r in list_r:
                linestyle = '-' if r == 4 else '--'
                plt.plot(xx, output[r][first:last], colors[k], label='r={0}'.format(r), marker=markers[k],
                         linestyle=linestyle, markevery=10)
                k += 1
            plt.xlabel('Time')
            plt.ylabel('Misclassification rate')
            plt.legend()
            plt.xlim(first, last)
            plt.savefig(os.path.join(self.path, '{0}-r_sensitivity-{1}.eps'.format(self.dataset_name, i)), format='eps')
            plt.savefig(os.path.join(self.path, '{0}-r_sensitivity-{1}.png'.format(self.dataset_name, i)), format='png', dpi=200)

    def plot_sensitivity_noise(self, output):
        """

        :param output:
        :return:
        """

        mpl.rcParams['lines.linewidth'] = 1.0
        mpl.rcParams['lines.markersize'] = 4

        xx = [0, 10, 20, 30]
        plt.figure(1)
        plt.clf()
        colors = ['green', 'red', 'blue']
        markers = ['s', 'o', 'x']
        k = 0
        for key in output.keys():
            plt.plot(xx, output[key], colors[k], label=key, marker=markers[k], markevery=10)
            k += 1
        plt.xlabel('noise_level')
        plt.ylabel('Misclassification rate')
        plt.legend()
        # plt.xlim(first, last)
        plt.savefig(os.path.join(self.path, '{0}-noise_sensitivity.eps'.format(self.dataset_name)), format='eps')
        plt.savefig(os.path.join(self.path, '{0}-noise_sensitivity.png'.format(self.dataset_name)), format='png', dpi=200)

    @staticmethod
    def store_results(output_list, path):
        """

        :param output_list:
        :param path:
        :return:
        """
        with open(os.path.join(path, 'data.pkl'), 'wb') as f:
            pickle.dump(output_list, f)

    @staticmethod
    def load_results(path):
        """

        :return:
        """
        with open(os.path.join(path, 'data.pkl'), 'rb') as f:
            outputs = pickle.load(f)
        return outputs

    @staticmethod
    def plot_training(output:dict, dataset_name, b_in, path, computation, algorithms=['Aware', 'MDDM', 'AUE', 'DSURF']):
        """

        :param output:
        :param dataset_name:
        :param b_in:
        :param path:
        :param computation:
        :param algorithms:
        :return:
        """

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

    @staticmethod
    def average_over_time(output_list):
        """

        :param output_list:
        :return:
        """
        output = {}

        if len(output_list) == 0:
            print('Error: no result')
        else:
            ave_over_time = {}
            var_over_time = {}
            for key in output_list[0].keys():
                b = len(output_list[0][key])
                output[key] = [0] * b

                for t in range(b):
                    output[key][t] = numpy.mean([o[key][t] for o in output_list])

                ave_over_time[key] = numpy.mean(output[key])
                var_over_time[key] = numpy.var(output[key])
                print('average over time {0} : {1}'.format(key, ave_over_time[key]))
                print('variance over time {0} : {1}'.format(key, var_over_time[key]))
                logging.info('average over time {0} : {1}'.format(key, ave_over_time[key]))
                logging.info('variance over time {0} : {1}'.format(key, var_over_time[key]))

        return ave_over_time, var_over_time

    @staticmethod
    def create_folder():
        """

        :return:
        """
        current_time = time.strftime('%Y-%m-%d_%H%M%S')

        path = os.path.join('output', current_time)
        os.makedirs(path)

        return path
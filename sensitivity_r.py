import models
import training
import results
import sys
import read_data as data
import hyperparameters
import random
import numpy
import logging


class Sensitivity_r:

    def __init__(self, dataset, list_r=[1,2,4,8,16]):

        self.dataset_name = dataset
        self.loss = {}
        self.list_r = list_r

    def process(self, delta=0.1, loss_fn='reg'):

        train = training.Training(self.dataset_name, algo_names=['DriftSurf'])
        outputs = {}
        for r in self.list_r:
            outputs[r] = train.process(delta, loss_fn, r=r)['DriftSurf']
        return outputs


# -------------- Sensitivity_r ------------------
if __name__ == "__main__":

    # if len(sys.argv) < 2:
    #     print("needs to enter dataset: ", ", ".join(data.read_dataset.AVAILABLE_DATASETS))
    #     exit()
    # dataset_name = sys.argv[1]

    list_r = [1,2,4,8,16]
    ave = {}

    path = results.Results.create_folder()

    for dataset_name in data.read_dataset.AVAILABLE_DATASETS:
        ave[dataset_name] = []

        sens = Sensitivity_r(dataset_name, list_r)

        outputs = sens.process(loss_fn='zero-one')
        results.Results.plot_sensitivity_r(dataset_name, outputs, list_r, path)
        for r in list_r:
            ave[dataset_name].append(numpy.mean(outputs[r]))
            # print('average over time for r = {0} : {1}'.format(r, numpy.mean(outputs[r])))
            # logging.info('average over time for r = {0} : {1}'.format(r, numpy.mean(outputs[r])))

    results.Results.store_results(ave, path)
    results.Results.plot_sensitivity_r_all(ave, list_r, path)
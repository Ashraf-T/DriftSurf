import models
import training
import results
import sys
import read_data as data
import hyperparameters
import random
import numpy
import logging


class greedy:

    def __init__(self, dataset):

        self.dataset_name = dataset
        self.loss = {}

    def process(self, delta=0.1, loss_fn='reg'):

        train = training.Training(self.dataset_name, algo_names=['DriftSurf_v2'])
        outputs = {}
        for method in [models.DriftSurf_v2.GREEDY, 'no-Greedy']:
            outputs[method] = train.process(delta, loss_fn, reactive_method=method)[0]['DriftSurf_v2']
        return outputs

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("needs to enter dataset: ", ", ".join(data.read_dataset.AVAILABLE_DATASETS))
        exit()
    dataset_name = sys.argv[1]

    path = results.Results.create_folder()

    expt = greedy(dataset_name)

    outputs = expt.process()
    results.Results.store_results(outputs,path)
    results.Results.plot_greedy(dataset_name, outputs, path)
    for method in outputs.keys():
        print('average over time for {0} : {1}'.format(method, numpy.mean(outputs[method])))
        logging.info('average over time for {0} : {1}'.format(method, numpy.mean(outputs[method])))

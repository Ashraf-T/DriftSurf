import models
import Training
import read_data as data
import hyperparameters
import random


class Sensitivity_r:

    # OPT = models.Opt.STRSAGA
    def __init__(self, dataset, list_r=[1,5,10]):

        self.dataset_name = dataset
        self.loss = {}
        self.list_r = list_r

    def process(self, delta=0.1, loss_fn='reg'):

        train = Training.Training(self.dataset_name, algo_names=['DSURF'])
        outputs = {}
        for r in self.list_r:
            outputs[r] = train.process(delta, loss_fn, r=r)['DSURF']
        return outputs


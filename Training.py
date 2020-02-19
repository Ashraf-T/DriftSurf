import models
import read_data as data
import hyperparameters
from drift_detection.__init__ import *
import random
import numpy
import logging


class Training:

    STRSAGA = models.Opt.STRSAGA
    SGD = models.Opt.SGD
    LIMITED = 'limited'
    BEFORE = 'B'
    AFTER = 'A'

    def __init__(self, dataset, computation='unlimited', rate=2, base_learner = 'STRSAGA', algo_names=['Aware', 'SGD', 'OBL', 'MDDM', 'AUE', 'DSURF']):
        """

        :param dataset:
        :param computation:
        :param rate:
        :param base_learner:
        :param algo_names:
        """
        self.algorithms = algo_names
        self.loss = {}
        self.computation = computation
        self.rate = rate

        self.opt = getattr(Training, base_learner)

        self.dataset_name = dataset
        read_data = data.read_dataset()
        self.X, self.Y, self.n, self.d, self.drift_times = read_data.read(self.dataset_name)

        if self.dataset_name.startswith('sea'):
            self.dataset_name = 'sea'

        self.mu = hyperparameters.MU[self.dataset_name]
        self.step_size = hyperparameters.STEP_SIZE[self.dataset_name]
        self.b = hyperparameters.b[self.dataset_name] if self.dataset_name in hyperparameters.b.keys() else hyperparameters.b['default']
        self.lam = int(self.n//self.b)
        self.rho = self.lam * self.rate

    @staticmethod
    def load_dataset(dataset):

        name = dataset
        if dataset.startswith('sea'):
            name = 'sea'

        read_data = data.read_dataset()
        X, Y, n, d, drift_times = read_data.read(dataset)
        return name, X, Y, n, d, drift_times

    @staticmethod
    def setup_experiment(dataset, rate, n):

        mu = hyperparameters.MU[dataset]
        step_size = hyperparameters.STEP_SIZE[dataset]
        b = hyperparameters.b[dataset] if dataset in hyperparameters.b.keys() else hyperparameters.b['default']
        lam = int(n//b)
        rho = lam * rate

        return mu, step_size, b, lam, rho

    def update_loss(self, test_set, time):
        """

        :param test_set:
        :param time:
        :return:
        """

        for algo in self.algorithms:
            self.loss[algo][time] = getattr(self, algo).zero_one_loss(test_set)

    def setup_algorithms(self, delta, loss_fn, detector=MDDM_G(), Aware_reset = 'B', r=None):
        """

        :param delta:
        :param loss_fn:
        :param detector:
        :param Aware_reset:
        :return:
        """

        for algo in self.algorithms:
            if algo == 'DSURF': self.setup_DSURF(delta, loss_fn, r)
            elif algo == 'MDDM': self.setup_MDDM(detector)
            elif algo == 'Aware': self.setup_Aware(Aware_reset)
            else: getattr(self, 'setup_{0}'.format(algo))()

    def setup_DSURF(self, delta, loss_fn, r=None):
        """

        :param delta:
        :param loss_fn:
        :return:
        """

        self.DSURF_t = 0
        self.DSURF_r = r if r else (hyperparameters.r[self.dataset_name] if self.dataset_name in hyperparameters.r.keys() else hyperparameters.r['default'])
        self.DSURF = models.LogisticRegression_DSURF(self.d, self.opt, delta, loss_fn)

    def setup_MDDM(self, detector=MDDM_G()):
        """

        :param detector:
        :return:
        """
        self.MDDM = models.LogisticRegression_expert(numpy.random.rand(self.d), self.opt)
        self.MDDM_drift_detector = detector

    def setup_AUE(self):
        """

        :return:
        """
        self.AUE = models.LogisticRegression_AUE(self.d, self.opt)

    def setup_Aware(self, reset='B'):
        """

        :param reset:
        :return:
        """
        self.Aware_reset = reset
        self.Aware = models.LogisticRegression_expert(numpy.random.rand(self.d), self.opt, self.S)

    def setup_SGD(self):
        """

        :return:
        """
        self.SGD = models.LogisticRegression_expert(numpy.random.rand(self.d), Training.SGD)

    def setup_OBL(self):
        """

        :return:
        """
        self.OBL = models.LogisticRegression_expert(numpy.random.rand(self.d), Training.STRSAGA)

    def update_strsaga_model(self, model):
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

    def update_sgd_model(self, model):
        """

        :param model:
        :return:
        """
        if model:
            sgdOnline_T = self.S
            for s in range(min(self.lam, self.rho)):
                if sgdOnline_T < self.S + self.lam:
                    j = sgdOnline_T
                    sgdOnline_T += 1
                point = (j, self.X[j], self.Y[j])
                model.update_step(point, self.step_size, self.mu)

    def process_MDDM(self, time, new_batch):
        """

        :param time:
        :param new_batch:
        :return:
        """

        if (self.MDDM_drift_detector.test(self.MDDM, new_batch) and time != 0):
            self.MDDM = models.LogisticRegression_expert(numpy.random.rand(self.d), self.opt, self.S)
            self.MDDM_drift_detector.reset()
            logging.info('MDDM drift detected, reset model : {0}'.format(time))

        getattr(self, 'update_{0}_model'.format(self.opt))(self.MDDM)
        # self.update_strsaga_model(self.MDDM)

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
            # self.update_strsaga_model(expert)
            getattr(self, 'update_{0}_model'.format(self.opt))(expert)

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
                getattr(self, 'update_{0}_model'.format(self.opt))(self.DSURF.expert_predictive)
                getattr(self, 'update_{0}_model'.format(self.opt))(self.DSURF.expert_stable)
                # self.update_strsaga_model(self.DSURF.expert_predictive)
                # self.update_strsaga_model(self.DSURF.expert_stable)

        if not self.DSURF.stable:
            # update models
            self.DSURF.update_reactive_sample_set(new_batch)
            getattr(self, 'update_{0}_model'.format(self.opt))(self.DSURF.expert_predictive)
            getattr(self, 'update_{0}_model'.format(self.opt))(self.DSURF.expert_reactive)
            # self.update_strsaga_model(self.DSURF.expert_predictive)
            # self.update_strsaga_model(self.DSURF.expert_reactive)

            self.DSURF_t += 1
            if self.DSURF_t == self.DSURF_r :
                self.DSURF.exit_reactive(self.S+self.lam, self.mu)

    def process_Aware(self):
        """

        :return:
        """
        # self.update_strsaga_model(self.Aware)
        getattr(self, 'update_{0}_model'.format(self.opt))(self.Aware)

    def process_OBL(self):
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
        self.update_sgd_model(self.SGD)

    def process(self, delta=0.2, loss_fn='zero-one', drift_detectr=MDDM_G(), Aware_reset='B', r=None):
        """

        :param delta:
        :param loss_fn:
        :param drift_detectr:
        :param Aware_reset:
        :return:
        """

        self.S = 0
        # self.rho = int(self.lam * rate)
        self.setup_algorithms(delta, loss_fn, drift_detectr, Aware_reset, r)

        for algo in self.algorithms:
            self.loss[algo] = [0] * self.b

        logging.info('dataset : {0}, n : {1}, b : {2}'.format(self.dataset_name, self.n, self.b))


        for time in range(self.b):

            print(time)

            if time in self.drift_times and 'Aware' in self.algorithms and self.Aware_reset == Training.BEFORE:
                self.setup_Aware()

            # measure accuracy over upcoming batch
            test_set = [(i, self.X[i], self.Y[i]) for i in range(self.S, self.S + self.lam)]
            self.update_loss(test_set, time)

            if time in self.drift_times and 'Aware' in self.algorithms and self.Aware_reset == Training.AFTER:
                self.setup_Aware()

            for algo in self.algorithms:
                if algo in ['SGD', 'OBL', 'Aware']: getattr(self, 'process_{0}'.format(algo))()
                else: getattr(self, 'process_{0}'.format(algo))(time, test_set)


            self.S += self.lam

        return self.loss


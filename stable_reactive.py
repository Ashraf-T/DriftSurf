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
import read_data as data
import logging

OPT = models.Opt.SAGA
OPT_sgd = models.Opt.SGD

T_reactive = 4
Delta = 0.1

factor = 1
NUMBER_OF_BATCHES = 50 # air: 10000, elec:50

STEP_SIZE = {'rcv': 5e-1, 'covtype': 5e-3, 'a9a': 5e-3, 'lux' : 5e-2, 'pow': 2e-2, 'air': 2e-2, 'elec': 2e-1}
MU        = {'rcv': 1e-5, 'covtype': 1e-4, 'a9a': 1e-3, 'lux' : 1e-3, 'pow': 1e-3, 'air': 1e-3, 'elec' : 1e-5}
THRESHOLD = {'default': 0.5}

# Drift_Times = {'lux': [], 'pow': [17, 47, 76], 'air': [32, 64, 100, 165, 462, 562, 687, 1010, 1042, 1100, 1385, 1582, 1682, 1720,1847], 'elec': [] }
Drift_Times = {'lux': [], 'pow': [17, 47, 76], 'air': [1682, 1720,1847], 'elec': [] }

def fresh_model(d, opt= OPT):
    return models.LogisticRegression_expert(numpy.random.rand(d), opt)


def fresh_StableReactive_model(d, OPT, current_pointer = 0, delta = Delta, loss_fn = 'reg'):
    return models.LogisticRegression_StableReactive(numpy.random.rand(d), OPT, current_pointer, delta, loss_fn)


"""
X, Y: data
n: number of points
d: dimension of data
b: length of stream
lam: lambda, constant arrivals
W: window size, in multiples of lambda
"""


def process(X, Y, n, d, step_size, mu, b, lam, rho, loss_fn, dataset_name):

    loss = {
        'Aware': {'zero-one': [0] * b, 'reg': [0]*b},
        'STR': {'zero-one': [0] * b, 'reg': [0]*b},
        'sgdOnline': {'zero-one': [0] * b, 'reg': [0]*b},
        'SR' : {'zero-one':[0]*b, 'reg':[0]*b}
    }

    aware = fresh_model(d)

    aware_T0 = 0
    str_T0 = 0
    aware_T1 = 0
    str_T1 = 0

    sgd = fresh_model(d, OPT_sgd)
    sr = fresh_StableReactive_model(d, OPT, loss_fn=loss_fn)
    STR = fresh_model(d, OPT)

    sr_t = 0
    S = 0  # S_i is [0:S]

    for time in range(b):

        print(time)

        if time in Drift_Times[dataset_name]:
                aware_T0 = lam * time
                aware_T1 = aware_T0

        # measure accuracy over upcoming batch
        test_set = [(i, X[i], Y[i]) for i in range(S, S + lam)]
        loss['Aware']['zero-one'][time] = aware.zero_one_loss(test_set)
        loss['sgdOnline']['zero-one'][time] = sgd.zero_one_loss(test_set)
        loss['SR']['zero-one'][time] = sr.zero_one_loss(test_set)
        loss['STR']['zero-one'][time] = STR.zero_one_loss(test_set)

        loss['Aware']['reg'][time] = aware.reg_loss(test_set, mu)
        loss['sgdOnline']['reg'][time] = sgd.reg_loss(test_set, mu)
        loss['SR']['reg'][time] = sr.reg_loss(test_set, mu)
        loss['STR']['reg'][time] = STR.reg_loss(test_set, mu)


        S_prev = S
        S += lam

        # SR algo process
        sr.update_perf_all(data = test_set, mu=mu)
        sr.update_best_observed_perf()
        if sr.enter_reactive(current_pointer=time*lam, data=test_set):
            logging.info('enter reactive state : {0}'.format(time))
        if sr.get_stable():
            # update models
            for key in ['main', 'new_stable']:
                if sr.experts[key]['expert'] != None:
                    T_pointers, _, T_size = sr.experts[key]['expert'].expert_effective_set()
                    lst = list(T_pointers)
                    for s in range(int(rho)):
                        if s % 2 == 0 and lst[1] < S:
                            j = lst[1]
                            T_size += 1
                            lst[1] += 1
                        else:
                            j = random.randrange(lst[0], lst[1])
                        point = (j, X[j], Y[j])
                        sr.experts[key]['expert'].update_step(point, step_size, mu)
                    sr.experts[key]['expert'].update_effective_set(new_pointer=lst[1], new_T_size=T_size)
                    if sr.pointer_buff < lst[1]:
                        sr.update_buffer_pointer(lst[1])
        else:
            # print('enter reactive state at time {0}'.format(time))
            if sr_t != T_reactive:
            # update models
                sr.update_reactive_sample_set(data=test_set)
                for key in ['main', 'new_reactive']:
                    # print(sr.experts[key]['current_perf'])
                    T_pointers, _, T_size = sr.experts[key]['expert'].expert_effective_set()
                    lst = list(T_pointers)
                    for s in range(int(rho)):
                        if s % 2 == 0 and lst[1] < S:
                            j = lst[1]
                            T_size += 1
                            lst[1] += 1
                        else:
                            j = random.randrange(lst[0], lst[1])
                        point = (j, X[j], Y[j])
                        sr.experts[key]['expert'].update_step(point, step_size, mu)
                    sr.experts[key]['expert'].update_effective_set(new_pointer=lst[1], new_T_size=T_size)
                    if sr.pointer_buff < lst[1]:
                        sr.update_buffer_pointer(lst[1])
                sr_t += 1
            else:
                sr.exit_reactive(current_pointer=time*lam, data=test_set, mu=mu)
                sr_t = 0
                for key in ['main', 'new_stable']:
                    if sr.experts[key]['expert'] != None:
                        T_pointers, _, T_size = sr.experts[key]['expert'].expert_effective_set()
                        lst = list(T_pointers)
                        for s in range(int(rho)):
                            if s % 2 == 0 and lst[1] < S:
                                j = lst[1]
                                T_size += 1
                                lst[1] += 1
                            else:
                                j = random.randrange(lst[0], lst[1])
                            point = (j, X[j], Y[j])
                            sr.experts[key]['expert'].update_step(point, step_size, mu)
                        sr.experts[key]['expert'].update_effective_set(new_pointer=lst[1], new_T_size=T_size)
                        if sr.pointer_buff < lst[1]:
                            sr.update_buffer_pointer(lst[1])

        # sgdOnline
        sgdOnline_T = S_prev
        for s in range(min(lam, rho)):
            if sgdOnline_T < S:
                j = sgdOnline_T
                sgdOnline_T += 1
            point = (j, X[j], Y[j])
            sgd.update_step(point, step_size, mu)

        for s in range(rho):
            # Aware
            if s % 2 == 0 and aware_T1 < S:
                j = aware_T1
                aware_T1 += 1
            else:
                j = random.randrange(aware_T0, aware_T1)
            point = (j, X[j], Y[j])
            aware.update_step(point, step_size, mu)

        for s in range(rho):
            # STR
            if s % 2 == 0 and str_T1 < S:
                j = str_T1
                str_T1 += 1
            else:
                j = random.randrange(str_T0, str_T1)
            point = (j, X[j], Y[j])
            STR.update_step(point, step_size, mu)

    return loss

def plot(output, rate, b_in, dataset_name):

    current_time = time.strftime('%Y-%m-%d_%H%M%S')
    path_png = os.path.join('output', current_time, 'png')
    path_eps = os.path.join('output', current_time, 'eps')

    os.makedirs(path_png)
    os.makedirs(path_eps)

    mpl.rcParams['lines.linewidth'] = 1.0
    mpl.rcParams['lines.markersize'] = 4

    b = min(b_in, 100)
    print(b)
    for i in range(max(b_in//100, 1)):
        # output = output_in[100 * i : 100 (i + 1)]
        # t = factor
        t = 1
        xx = range(0, b)
        tt1 = range(int(0.5 * b), int(0.5 * b) + 20)
        tt2 = range(int(0.5 * b) * 2, int(0.5 * b) * 2 + 20)
        xxx = range(0, b, 10)
        xxxx = range(0, b, int(b*0.5))

        first = i*b
        last = b * (i+1)

        xx = range(first, last)

        if i == 0:
            xx = range(1,b)
            first = 1


        # ------------ accuracy  --------------
        plt.figure(1)
        plt.clf()
        plt.plot(xx, output['Aware']['zero-one'][first:last], 'g-', label='Aware')
        plt.plot(xx, output['STR']['zero-one'][first:last], 'lime', label='STR')
        plt.plot(xx, output['sgdOnline']['zero-one'][first:last], 'peru', label='sgdOnline')
        plt.plot(xx, output['SR']['zero-one'][first:last], 'yellow', label='SR')
        # for x in xxx: plt.axvline(x=x, color='0.4', linestyle=':', linewidth=1)
        # for x in xxxx: plt.axvline(x=x, color='0.2', linestyle='--', linewidth=1)
        plt.xlabel('Time')
        plt.ylabel('Misclassification rate')
        plt.legend()
        plt.xlim(first, last)
        # plt.ylim(0.2, 0.7)
        # plt.savefig(os.path.join(path_eps, '{0}r{1}-acc{2}.eps'.format(dataset_name, rate, i)), format='eps')
        plt.savefig(os.path.join(path_png, '{0}r{1}-acc{2}.png'.format(dataset_name, rate, i)), format='png', dpi=200)


        # ------------ reg-loss  --------------
        plt.figure(2)
        plt.clf()
        plt.plot(xx, output['Aware']['reg'][first:last], 'g-', label='Aware')
        plt.plot(xx, output['STR']['reg'][first:last], 'lime', label='STR')
        plt.plot(xx, output['sgdOnline']['reg'][first:last], 'peru', label='sgdOnline')
        plt.plot(xx, output['SR']['reg'][first:last], 'yellow', label='SR')
        # for x in xxx: plt.axvline(x=x, color='0.4', linestyle=':', linewidth=1)
        # for x in xxxx: plt.axvline(x=x, color='0.2', linestyle='--', linewidth=1)
        plt.xlabel('Time')
        plt.ylabel('regression loss')
        plt.legend()
        plt.xlim(first, last)
        # plt.ylim(0.4, 0.9)
        # plt.savefig(os.path.join(path_eps, '{0}r{1}-reg{2}.eps'.format(dataset_name, rate, i)), format='eps')
        plt.savefig(os.path.join(path_png, '{0}r{1}-reg{2}.png'.format(dataset_name, rate, i)), format='png', dpi=200)


def median_outputs(output_list, b):
    output = {
        'Aware': {'zero-one': [0] * b, 'reg': [0] * b},
        'STR': {'zero-one': [0] * b, 'reg': [0] * b},
        'SR': {'zero-one': [0] * b, 'reg': [0] * b},
        'sgdOnline': {'zero-one': [0] * b, 'reg': [0] * b}
    }

    for t in range(b):
        output['Aware']['zero-one'][t] = numpy.median([o['Aware']['zero-one'][t] for o in output_list])
        output['STR']['zero-one'][t] = numpy.median([o['STR']['zero-one'][t] for o in output_list])
        output['SR']['zero-one'][t] = numpy.median([o['SR']['zero-one'][t] for o in output_list])
        output['sgdOnline']['zero-one'][t] = numpy.median([o['sgdOnline']['zero-one'][t] for o in output_list])

        output['Aware']['reg'][t] = numpy.median([o['Aware']['reg'][t] for o in output_list])
        output['STR']['reg'][t] = numpy.median([o['STR']['reg'][t] for o in output_list])
        output['SR']['reg'][t] = numpy.median([o['SR']['reg'][t] for o in output_list])
        output['sgdOnline']['reg'][t] = numpy.median([o['sgdOnline']['reg'][t] for o in output_list])

    return output

if __name__ == "__main__":

    dataset_name = 'air'
    prediction_threshold = THRESHOLD['default']
    b = NUMBER_OF_BATCHES

    rate = 2
    step_size = STEP_SIZE[dataset_name]
    mu = MU[dataset_name]

    # loss_fn = 'zero-one'
    loss_fn = 'reg'

    logging.basicConfig(filename='{0}-{1}-b{2}.log'.format(dataset_name, loss_fn, b), filemode='w', level=logging.INFO)

    # X, Y, n, d = data.Luxembourgcal()
    # X, Y, n, d = data.powerSupply()
    # X, Y, n, d = data.airline()
    X, Y, n, d = data.airline_trim(1900*581, 1700*581)
    # X, Y, n, d = data.elec()

    logging.info('Started')
    logging.info('dataset : {0}, n : {1}, b : {2}, T : {3}, delta : {4}'.format(dataset_name, n, b, T_reactive, Delta))

    lam = n // b
    rho = int(lam * rate)
    N = 1
    outputs = []
    for i in range(N):
#        print ({'Trial {0}'.format(i)})
        logging.info('Trial {0}'.format(i))
        output = process(X, Y, n, d, step_size, mu, b, lam, rho, loss_fn, dataset_name)
        outputs.append(output)

    with open('output/data/{0}r{1}T{2}-{3}-b{4}d{5}.pkl'.format(dataset_name, rate, T_reactive, loss_fn, b, Delta), 'wb') as f:
        pickle.dump(outputs, f)

    with open('output/data/{0}r{1}T{2}-{3}-b{4}d{5}.pkl'.format(dataset_name, rate, T_reactive, loss_fn, b, Delta), 'rb') as f:
        outputs = pickle.load(f)

    output = median_outputs(outputs, b)

    plot(output, rate, b, dataset_name)

import csv
import math

class read_dataset:

    AVAILABLE_DATASETS = ['sine1', 'sea0', 'sea10', 'sea20', 'sea30', 'sea_gradual', 'mixed', 'Circles', 'hyperplane_slow', 'hyperplane_fast',
                          'powersupply', 'elec', 'airline', 'rcv', 'covtype']

    def read(self, dataset_name):

        if dataset_name.startswith('sea'):
            if dataset_name == 'sea_gradual':
                return getattr(self, dataset_name)()
            else:
                name = 'sea'
                noise_level = dataset_name[3:]
                return getattr(self, name)(noise_level)

        elif dataset_name.startswith('hyperplane'):
            name = 'hyperplane'
            type = dataset_name[11:]
            return getattr(self, name)(type)

        else:
            return getattr(self, dataset_name)()

    # synthetic datasets :
    def sine1(self):
        """ read synthetic dataset sine1 which is generated using Tornado framework
            'Pesaranghader, A. and Viktor, H. L.  Fast hoeffding driftdetection method for evolving data streams.  InECMLPKDD, pp. 96–111, 2016'

        :return:
            features, labels, #records, #features, times that drift happen
        """
        n = 10000
        d = 2 + 1
        drift_times = [20,40,60,80]

        X = []
        Y = []

        with open('data/synthetic/sine1.arff') as file:
            i = 0

            for line in file:
                if i > 5:
                    fields = line.strip().split(',')
                    label = 1 if (fields[len(fields) - 1]) == 'p' else -1
                    features = {0: 1}
                    for j in range(len(fields) - 1):
                        features[j + 1] = float(fields[j])

                    X.append(features)
                    Y.append(label)
                i += 1
        assert len(X) == n

        return X, Y, n, d, drift_times

    def mixed(self):
        """
        :return:
                features, labels, #records, #features, times that drift happen
        """
        n = 100000
        d = 4 + 1
        drift_times = [20, 40, 60, 80]

        X = []
        Y = []

        with open('data/synthetic/MIXED.arff') as file:
            reader = csv.reader(file, delimiter=',')
            i = 0
            for line in reader:
                features = {}
                features[0] = 1 if line[0] == 'True' else 0
                features[1] = 1 if line[1] == 'True' else 0
                for j in range(2, d - 1):
                    features[j] = float(line[j])
                features[d - 1] = 1
                label = 1 if line[d - 1] == 'p' else -1

                X.append(features)
                Y.append(label)
                i += 1
        assert len(X) == n

        return X, Y, n, d, drift_times

    def sea(self, noise_level=0):
        """ read synthetic dataset sea which is generated using MOA framework
            'Bifet, A., Holmes, G., Kirkby, R., and Pfahringer, B. Moa:Massive online analysis.JMLR, 11:1601–1604, 2010.'

        :param noise_level:
                level of noise added to the dataset
        :return:
                features, labels, #records, #features, times that drift happen
        """
        n = 100000
        d = 3 + 1
        drift_times = [25, 50, 75]

        X = []
        Y = []

        with open('data/synthetic/sea_abrupt{0}.arff'.format(noise_level)) as file:
            reader = csv.reader(file, delimiter=',')
            i = 0
            for line in reader:
                features = {}
                for j in range(d - 1):
                    features[j] = float(line[j])
                features[d - 1] = 1
                label = 1 if line[d - 1] == 'groupA' else -1

                X.append(features)
                Y.append(label)
                i += 1
        assert len(X) == n

        return X, Y, n, d, drift_times

    def sea_gradual(self):
        """ read synthetic dataset sea which is generated using MOA framework
            'Bifet, A., Holmes, G., Kirkby, R., and Pfahringer, B. Moa:Massive online analysis.JMLR, 11:1601–1604, 2010.'

        :return:
                features, labels, #records, #features, times that drift happen
        """
        n = 100000
        d = 3 + 1
        drift_times = [40, 60]

        X = []
        Y = []

        with open('data/synthetic/sea_gradual.arff') as file:
            reader = csv.reader(file, delimiter=',')
            i = 0
            for line in reader:
                features = {}
                for j in range(d - 1):
                    features[j] = float(line[j])
                features[d - 1] = 1
                label = 1 if line[d - 1] == 'groupA' else -1

                X.append(features)
                Y.append(label)
                i += 1
        assert len(X) == n

        return X, Y, n, d, drift_times

    def circles(self):
        """ read synthetic dataset sea which is generated using MOA framework
            'Bifet, A., Holmes, G., Kirkby, R., and Pfahringer, B. Moa:Massive online analysis.JMLR, 11:1601–1604, 2010.'

        :return:
                features, labels, #records, #features, times that drift happen
        """
        n = 10000
        d = 2 + 1
        drift_times = [25, 30, 50, 55, 75, 80]

        X = []
        Y = []

        with open('data/synthetic/circles.arff') as file:
            reader = csv.reader(file, delimiter=',')
            i = 0
            for line in reader:
                features = {}
                for j in range(d - 1):
                    features[j] = float(line[j])
                features[d - 1] = 1
                label = 1 if line[d - 1] == 'p' else -1

                X.append(features)
                Y.append(label)
                i += 1
        assert len(X) == n

        return X, Y, n, d, drift_times

    def hyperplane(self, type='slow'):
        """ read synthetic dataset hyperplane which is generated using MOA framework
            'Bifet, A., Holmes, G., Kirkby, R., and Pfahringer, B. Moa:Massive online analysis.JMLR, 11:1601–1604, 2010.'

        :param type: (slow / fast)
        :return:
                features, labels, #records, #features, times that drift happen
        """
        n = 100000
        d = 10 + 1

        X = []
        Y = []

        with open('data/synthetic/hyperplane_{0}.arff'.format(type)) as file:
            reader = csv.reader(file, delimiter=',')
            i = 0
            for line in reader:
                features = {}
                for j in range(d - 1):
                    features[j] = float(line[j])
                features[d - 1] = 1
                label = 1 if line[d - 1] == 'class1' else -1

                X.append(features)
                Y.append(label)
                i += 1
        assert len(X) == 100000

        return X, Y, n, d, []

    # real world datasets :
    def powersupply(self):
        """ read readl-world dataset powersupply
            'Dau, H. A., Bagnall, A., Kamgar, K., Yeh, C.-C. M., Zhu,Y., Gharghabi, S., Ratanamahatana, C. A., and Keogh,E.  The ucr time series archive.IEEE/CAA Journal ofAutomatica Sinica, 6(6):1293–1305, 2019.'

        :return:
                features, labels, #records, #features, times that drift happen
        """
        n = 29928
        d = 2 + 1

        drift_times = [17, 47, 76]

        X = []
        Y = []

        with open('data/real-world/powersupply.arff') as file:
            i = 0
            for line in file:
                fields = line.strip().split(',')
                label = 1 if int(fields[2]) < 12 else -1
                features = {0: 1}
                for j in range(2):
                    features[j+1] = float(fields[j])
                X.append(features)
                Y.append(label)
                i += 1
        assert len(X) == n

        max_0 = max(X[i][1] for i in range(len(X)))
        max_1 = max(X[i][2] for i in range(len(X)))
        for i in range(len(X)):
            X[i][1] = X[i][1]/max_0
            X[i][2] = X[i][2]/max_1


        return X, Y, n, d, drift_times

    def elec(self):
        """ read real-world dataset electricity
            'Harries, M., cse tr, U. N., and Wales, N. S.  Splice-2 com-parative evaluation: Electricity pricing. Technical report,1999'

        :return:
            features, labels, #records, #features, times that drift happen
        """

        #   features: date, day, period, nswprice, nswdemand, vicprice, vicdemand, transfer

        n = 45312
        d = 13 + 1

        drift_times = [20]

        X = []
        Y = []
        with open('data/real-world/electricity-normalized.csv') as file:
            i = 0
            for line in file:
                fields = line.strip().split(',')
                label = 1 if fields[len(fields)-1] == 'UP' else -1

                features = {0:1}
                features[int(fields[1])] = 1
                for j in range(2, len(fields)-1):
                    features[j+6] = float(fields[j])
                X.append(features)
                Y.append(label)
                i += 1

        assert len(X) == n

        return X, Y, n, d, drift_times

    def airline(self, num2=58100, num1=0):
        """ read real-world dataset airline
        Note1. We preprocessed this dataset by normalization and one-hot encoding of categorical features)
        Note2. The first 100 batch are used in this experimnets)
        'Ikonomovska, E. Airline dataset.http://kt.ijs.si/elena_ikonomovska/data.html.  (Accessed on02/06/2020).'

        :param num2:
        :param num1:
        :return:
                features, labels, #records, #features, times that drift happen
        """
        # n = 5810462
        n = num2 - num1 # read a partial of this dataset
        d = 679 #there are only 13 features in the original dataset. One-hot encoding will results in having dimension of 679

        drift_times = [31, 67]

        X = []
        Y = []
        with open('data/real-world/airline.data') as file:
            i = 0
            for line in file:
                if  num1 <= i < num2:
                    fields = line.strip().split(',')
                    label = int(fields[len(fields)-1])

                    features = {0:1}
                    for j in range(len(fields)-1):
                        (index, val) = fields[j].split(':')
                        features[int(index)] = float(val)
                    X.append(features)
                    Y.append(label)
                i += 1
        assert len(X) == n

        return X, Y, n, d, drift_times

    # semi-synthetic datasets :
    def rcv(self, drift_portion=0.3):
        """ read dataset rcv and introduce two abrupt drifts to it

        :param drift_portion:
        :return:
                features, labels, #records, #features, times that drift happen
        """
        n = 20242
        d = 47237
        drift_times = [30, 60]

        X = []
        Y = []

        with open('data/semi-synthetic/rcv_shuf.data') as file:
            i = 0
            for line in file:
                fields = line.strip().split(' ')
                label = int(fields[0])
                features = {0: 1}
                for j in range(1, len(fields)):
                    (index, val) = fields[j].split(':')
                    features[int(index)] = float(val)
                if n * drift_portion <= i < 2 * n * drift_portion:
                        label *= -1
                X.append(features)
                Y.append(label)
                i += 1
        assert len(X) == n

        return X, Y, n, d, drift_times

    def covtype(self, drift_portion=0.3):
        """ read dataset covertype and introduce two abrupt drifts to it

        :param drift_portion:
        :return:
            features, labels, #records, #features, times that drift happen
        """
        n = 581012
        d = 54 + 1
        drift_times = [30, 60]

        X = []
        Y = []

        with open('data/semi-synthetic/covtype_shuf.data') as file:
            i = 0
            for line in file:
                fields = line.strip().split(' ')
                label = 1 if int(fields[0]) == 1 else -1
                features = {0: 1}
                for j in range(1, len(fields)):
                    (index, val) = fields[j].split(':')
                    features[int(index)] = float(val)
                X.append(features)
                Y.append(label)
                i += 1
        assert len(X) == n

        # for a drift of rate 0.4
        (i, j, theta) = (1, 8, 3.14)
        Rx = X[:]
        for k in range(len(Rx)):
            if len(Rx) * drift_portion <= k < 2 * len(Rx) * drift_portion:
                x = X[k].copy()
                xi = x[i] if i in x else 0
                xj = x[j] if j in x else 0
                x[i] = xi * math.cos(theta) - xj * math.sin(theta)
                x[j] = xi * math.sin(theta) + xj * math.cos(theta)
                Rx[k] = x
        X = Rx[:]

        return X, Y, n, d, drift_times
import csv
import math

class read_dataset_noDrift:

    AVAILABLE_DATASETS = ['sine1', 'sea0', 'sea10', 'sea20', 'sea30', 'mixed', 'Circles', 'hyperplane',
                          'rcv', 'covtype']

    def read(self, dataset_name):

        if dataset_name.startswith('sea'):
            if dataset_name == 'sea_gradual':
                return getattr(self, dataset_name)()
            else:
                name = 'sea'
                noise_level = dataset_name[3:]
                return getattr(self, name)(noise_level)

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
        drift_times = []

        X = []
        Y = []

        with open('data/synthetic/noDrift/sine1_noDrift.arff') as file:
            i = 0

            for line in file:
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
        drift_times = []

        X = []
        Y = []

        with open('data/synthetic/noDrift/mixed_noDrift.arff') as file:
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
        drift_times = []

        X = []
        Y = []

        with open('data/synthetic/noDrift/sea_nodrift{0}.arff'.format(noise_level)) as file:
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
        drift_times = []

        X = []
        Y = []

        with open('data/synthetic/noDrift/circles_noDrift.arff') as file:
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

    def hyperplane(self):
        """ read synthetic dataset hyperplane which is generated using MOA framework
            'Bifet, A., Holmes, G., Kirkby, R., and Pfahringer, B. Moa:Massive online analysis.JMLR, 11:1601–1604, 2010.'

        :return:
                features, labels, #records, #features, times that drift happen
        """
        n = 100000
        d = 10 + 1

        X = []
        Y = []

        with open('data/synthetic/noDrift/hyperplane-nodrift.arff') as file:
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

    # semi-synthetic datasets :
    def rcv(self):
        """ read dataset rcv and introduce two abrupt drifts to it

        :return:
                features, labels, #records, #features, times that drift happen
        """
        n = 20242
        d = 47237
        drift_times = []

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
                X.append(features)
                Y.append(label)
                i += 1
        assert len(X) == n

        return X, Y, n, d, drift_times

    def covtype(self):
        """ read dataset covertype and introduce two abrupt drifts to it

        :return:
            features, labels, #records, #features, times that drift happen
        """
        n = 581012
        d = 54 + 1
        drift_times = []

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

        return X, Y, n, d, drift_times
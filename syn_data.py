import csv

# """
# SEA, 1 abrupt drift at 50k; 10% noise added
# WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.SEAGenerator -f 3 -n 10 -b) -d (generators.SEAGenerator -f 2 -n 10 -b) -p 50000 -w 1) -f sea_abrupt.arff -m 100000 -h
# """
# def sea_abrupt():
    # n = 100000
    # d = 3+1
    # X = []
    # Y = []
    
    # with open('data/moa/sea_abrupt.arff') as file:
        # reader = csv.reader(file, delimiter=',')
        # i = 0
        # for line in reader:
            # features = {}
            # for j in xrange(d-1):
                # features[j] = float(line[j])
            # features[d-1] = 1
            # if line[d-1] == 'groupA':
                # label = 1
            # else:
                # label = -1
            # X.append(features)
            # Y.append(label)
            # i += 1
    # assert len(X) == 100000
            
    # return X, Y, n, d

"""
SEA, 3 abrupt drifts at 25k, 50k, 75k; 10% noise added
WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.SEAGenerator -f 3 -n 10 -b) -d 
    (ConceptDriftStream -s (generators.SEAGenerator -f 2 -n 10 -b) -d 
        (ConceptDriftStream -s (generators.SEAGenerator -f 4 -n 10 -b) -d 
            (generators.SEAGenerator -f 1 -n 10 -b)
        -p 25000 -w 1) 
    -p 25000 -w 1)
-p 25000 -w 1) -f sea4.arff -m 100000 -h
"""
def sea4():
    n = 100000
    d = 3+1
    X = []
    Y = []
    
    with open('data/moa/sea4.arff') as file:
        reader = csv.reader(file, delimiter=',')
        i = 0
        for line in reader:
            features = {}
            for j in xrange(d-1):
                features[j] = float(line[j])
            features[d-1] = 1
            if line[d-1] == 'groupA':
                label = 1
            else:
                label = -1
            X.append(features)
            Y.append(label)
            i += 1
    assert len(X) == 100000
            
    return X, Y, n, d
    
"""
STAGGER, 1 abrupt drift at 50k
WriteStreamToARFFFile -s (ConceptDriftStream -s (generators.STAGGERGenerator -f 3 -b) -d (generators.STAGGERGenerator -f 2 -b) -p 50000 -w 1) -f stagger.arff -m 100000 -h
"""
def stagger_abrupt():
    n = 100000
    d = 10
    X = []
    Y = []
    
    with open('data/moa/stagger.arff') as file:
        reader = csv.reader(file, delimiter=',')
        i = 0
        for line in reader:
            features = {0:1, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0, 9:0}
            #0 is const, 1-3 are size, 4-6 are color, 7-9 are shape
            
            size = line[0]
            color = line[1]
            shape = line[2]
            
            if size == 'small':
                features[1] = 1
            elif size == 'medium':
                features[2] = 1
            elif size == 'large':
                features[3] = 1
                
            if color == 'red':
                features[4] = 1
            elif color == 'blue':
                features[5] = 1
            elif color == 'green':
                features[6] = 1
            
            if shape == 'circle':
                features[7] = 1
            elif shape == 'square':
                features[8] = 1
            elif shape == 'triangle':
                features[9] = 1
            
            if line[3] == 'true':
                label = 1
            else:
                label = -1
                
            X.append(features)
            Y.append(label)
            i += 1
    assert len(X) == 100000
    
    return X, Y, n, d
    
"""
Incremental drift - Rotating Hyperplane Slow; 5% noise added; 10% prob of sign change
WriteStreamToARFFFile -s (generators.HyperplaneGenerator -k 10 -t 0.001 -s 10) -f hyperplane_slow.arff -m 100000 -h
"""
def hyperplane_slow():
    n = 100000
    d = 10 + 1
    X = []
    Y = []    

    with open('data/moa/hyperplane_slow.arff') as file:
        reader = csv.reader(file, delimiter=',')
        i = 0
        for line in reader:
            features = {}
            for j in xrange(d-1):
                features[j] = float(line[j])
            features[d-1] = 1
            if line[d-1] == 'class1':
                label = 1
            else:
                label = -1
            X.append(features)
            Y.append(label)
            i += 1
    assert len(X) == 100000
            
    return X, Y, n, d
    
"""
Incremental drift - Rotating Hyperplane Fast; 5% noise added; 10% prob of sign change
WriteStreamToARFFFile -s (generators.HyperplaneGenerator -k 10 -t 0.1 -s 10) -f hyperplane_fast.arff -m 100000 -h
"""
def hyperplane_fast():
    n = 100000
    d = 10 + 1
    X = []
    Y = []    

    with open('data/moa/hyperplane_fast.arff') as file:
        reader = csv.reader(file, delimiter=',')
        i = 0
        for line in reader:
            features = {}
            for j in xrange(d-1):
                features[j] = float(line[j])
            features[d-1] = 1
            if line[d-1] == 'class1':
                label = 1
            else:
                label = -1
            X.append(features)
            Y.append(label)
            i += 1
    assert len(X) == 100000
            
    return X, Y, n, d
    

# if __name__ == "__main__":
    # sea_abrupt()

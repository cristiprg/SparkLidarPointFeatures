from collections import Mapping, Container
from sys import getsizeof


def deep_getsizeof(o, ids):
    """Find the memory footprint of a Python object

    This is a recursive function that drills down a Python object graph
    like a dictionary holding nested dictionaries with lists of lists
    and tuples and sets.

    The sys.getsizeof function does a shallow size of only. It counts each
    object inside a container as pointer only regardless of how big it
    really is.

    :param o: the object
    :param ids:
    :return:
    """
    d = deep_getsizeof
    if id(o) in ids:
        return 0

    r = getsizeof(o)
    ids.add(id(o))

    if isinstance(o, str) or isinstance(0, unicode):
        return r

    if isinstance(o, Mapping):
        return r + sum(d(k, ids) + d(v, ids) for k, v in o.iteritems())

    if isinstance(o, Container):
        return r + sum(d(x, ids) for x in o)

    return r


from pyspark import SparkConf, SparkContext
from sklearn.neighbors import NearestNeighbors
from sklearn import decomposition
from scipy.stats import entropy

import pandas as pd
import numpy as np
import pickle
import time

#homeDir = '/home/hadoop/prigoana/data/'
homeDir = '/home/prigoana/data/'
csvFilePath = homeDir + 'bildstein_station1_xyz_intensity_rgb.txt'


def getData(refresh = False):
    if refresh:
        print "Reading data..."
        df = pd.read_csv(csvFilePath, usecols=[0, 1, 2], delimiter=' ')
        numpyData = df.as_matrix()
        print "Data read!"

        # Save data to disk
        with open(homeDir + 'bildstein_numpy_data.pkl', 'wb') as output:
            pickle.dump(numpyData, output, pickle.HIGHEST_PROTOCOL)

        return numpyData
    else:
        with open(homeDir + 'bildstein_numpy_data.pkl', 'rb') as input:
            return pickle.load(input)


def getKNNObject(refresh = False, size = -1):
    if refresh:
        numpyData = getData(refresh=refresh)[:size]

        print "Fitting..."
        start = time.time()
        knnobj = NearestNeighbors(n_neighbors=100, n_jobs=-1, algorithm='kd_tree', leaf_size=100).fit(numpyData)
        end = time.time()
        print "Fitted! " + str(end - start) + " seconds"

        # Save data to disk
        with open(homeDir + 'bildstein_knnobj.pkl', 'wb') as output:
            pickle.dump(knnobj, output, pickle.HIGHEST_PROTOCOL)

        return knnobj
    else:
        with open(homeDir + 'bildstein_knnobj.pkl', 'rb') as input:
            return pickle.load(input)


def mapFunction(iterator):
    partitionElements = list(iterator)
    print "Nr elements for worker " + str(len(partitionElements))
    print "Started inference on worker ..."
    ret_val = knnobj.kneighbors(partitionElements, return_distance=False)
    print "Finished inference on worker !!!"
    # print "ret_val = " + str(ret_val)

    for list_of_ids in ret_val:
        # 1. get the real point of the indices
        # 2. get PCA of the real points
        pca = decomposition.PCA()
        eignvalues = pca.fit([numpyData[id] for id in list_of_ids]).explained_variance_ratio_
        yield eignvalues, entropy(eignvalues)


    #return ret_val # TODO: vezi ce contine ret_val, si in loc sa returnezi asa, dai cu yield
    #yield list(iterator)



# Get only XYZ
#inputRDD = sc.textFile(csvFilePath).map(lambda x: [float(el) for el in x.split(' ')[0:3]])
#nrRows = inputRDD.count()

knnobj = getKNNObject(refresh=False, size=1000)
print "params = " + str(knnobj.get_params())


if True:
    sparkConf = SparkConf().setAppName("PythonLidarPointFeatures")#.setMaster('local[*]').set("spark.local.dir", "/media/cristiprg/Eu/tmp/Spark")
    sc = SparkContext(conf=sparkConf)

    print "Creating RDD ..."
    numpyData = getData(refresh=False)[:1000]
    inputRDD = sc.parallelize(numpyData, 6)
    print "RDD Created! Size = " + str(inputRDD.count())

    print "Performing inference on workers ..."

    results = inputRDD.mapPartitions(mapFunction)
    results.saveAsTextFile("/home/prigoana/data/SparkOutput")

    # "results" is now an RDD with numpy arrays containing the IDs of the nearest neighbors
    #results.foreach()


    sc.stop()

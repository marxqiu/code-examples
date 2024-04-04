from runner import run, run_special
from data import get_dataset
import numpy as np
from math import sqrt
import argparse
import sys; sys.path.append("/root/SPTAG/Release"); sys.path.append("/root/flann-master/src/python")




if __name__ == "__main__":
    
    datasets = [
    #"audio_192_euclidean",
    #"lastfm-64-dot",
    #"cifar_512_euclidean",
    #"millionSong_420_euclidean",
    #"deep-image-96-angular-1M",
    #"deep-image-96-angular-0.1M"
    #"mnist-784-euclidean",
    #"enron_1369_euclidean",
    #"notre_128_euclidean",
    #"fashion-mnist-784-euclidean",
    #"nuswide_500_euclidean",
    #"gist-960-euclidean",
    #"nytimes-256-angular",
    "sift-128-euclidean",
    #"glove-100-angular",
    #"sun_512_euclidean",
    #"glove-200-angular",
    #"trevi_4096_euclidean",
    #"glove-25-angular",
    #"ukbench_128_euclidean",
    #"glove-50-angular",
    #"imageNet_150_euclidean"
    ]
    
    
    for dataset in datasets:
    
        K = 3
        D, dimension = get_dataset(dataset)
        metric = D.attrs['distance']
        
        building_arguments = {"mode": 'BKT', 'metric' : metric, "dataset":dataset}
        query_arguments = [[2048, 2], [2048, 12], [2048, 20], [2048, 50]]
        algo = 'spann_ssd'
        run(algo, dataset, building_arguments, query_arguments, K)
        
        with open(dataset + '.txt', 'w') as f:
            f.write('Completed.')

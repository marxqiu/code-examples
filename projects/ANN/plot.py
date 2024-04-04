from result import load_all_results
from data import get_dataset
import matplotlib.pyplot as plt
import numpy as np

def recall(true_nn_distances, run_distances, times, properties):
    count = properties['count']
    recalls = np.zeros(len(run_distances))#for all test querys
    for i in range(len(run_distances)):
        t = true_nn_distances[i][count - 1] + 1e-3 
        #print(true_nn_distances[i][:count])
        #print("Test")
        #print(run_distances[i][:count])
        actual = 0
        for d in run_distances[i][:count]:
            if d <= t:
                actual += 1
        recalls[i] = actual
    return np.mean(recalls) / float(count)
            
def qps(true_nn_distances, run_distances, times, properties):
    return 1.0 / properties["best_search_time"]


metrics = {
    'recall' : recall,
    'qps' : qps

}


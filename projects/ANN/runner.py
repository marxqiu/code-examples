from data import get_dataset, dataset_transform
import numpy 
import psutil
from scipy.spatial.distance import pdist
from result import store_results
import time
import importlib
import json
import re

metrics = {
    'euclidean' : lambda a, b : pdist([a, b], metric='euclidean')[0],
    'angular' : lambda a, b : pdist([a, b], metric='cosine')[0]


}


def run(algo_name, dataset, building_arguments, query_arguments, k):
    #initialize the algorithm
    my_module = importlib.import_module("algorithms")
    module = getattr(my_module, algo_name)
    algo = module(**building_arguments)
    
    
    D, dimension = get_dataset(dataset)
    distance = D.attrs['distance']
    X_train = numpy.array(D['train'])
    X_test = numpy.array(D['test'])

    print('got a train set of size (%d * %d)' % (X_train.shape[0], dimension))
    print('got %d queries' % len(X_test))
    
    X_train, X_test = dataset_transform(D)
    
    t0 = time.time()
    memory_usage_before = psutil.Process().memory_info().rss / 1024
    algo.fit(X_train)
    build_time = time.time() - t0
    index_size = psutil.Process().memory_info().rss / 1024 - memory_usage_before
    print('Built index in', build_time)
    print('Index size: ', index_size)
    
    for id, query_argument in enumerate(query_arguments, 1):
        print("Running query argument %d of %d..." % (id, len(query_arguments)))
        algo.set_query_arguments(*query_argument)
        best_search_time = float('inf')
        
        for i in range(5):
            print('Run %d/5...' % (i + 1))
            n_items = 0
            results = []
            
            
            for x in X_test:
                start = time.time()
                candidates = algo.query(x, k)
                search_time = (time.time() - start)
                candidates = [(int(idx), float(metrics[distance](x, X_train[idx]))) \
                          for idx in candidates]#other distances
                n_items += 1
                
                if n_items % 1000 == 0:
                    print("Processed %d/%d queries..." % (n_items, len(X_test)))
            
                results.append((search_time, candidates))

                          
            total_search_time = sum(time for time, _ in results)
            total_candidates = sum(len(candidates) for _, candidates in results)
            search_time = total_search_time / len(X_test)
            avg_candidates = total_candidates / len(X_test)#和k有什么区别
            best_search_time = min(best_search_time, search_time)
        
        attrs = {
        "best_search_time": best_search_time,
        "candidates": avg_candidates,
        "name": str(algo),
        "count": int(k),
        "build_time": build_time,
        "index_size": index_size,
        "dataset": dataset,
        "algo": algo_name,
        "building_arguments": re.sub(r'\W+', '_', json.dumps(building_arguments, sort_keys=True)).strip('_'),
        "query_argument": re.sub(r'\W+', '_', json.dumps(query_argument, sort_keys=True)).strip('_')
        }
        
        store_results(dataset, k, attrs['query_argument'], attrs, results, attrs['building_arguments'])
        
    
def run_special(algo_name, dataset, building_arguments, k):
    #initialize the algorithm
    my_module = importlib.import_module("algorithms")
    module = getattr(my_module, algo_name)
    algo = module(**building_arguments)
    
    
    D, dimension = get_dataset(dataset)
    distance = D.attrs['distance']
    X_train = numpy.array(D['train'])
    X_test = numpy.array(D['test'])

    print('got a train set of size (%d * %d)' % (X_train.shape[0], dimension))
    print('got %d queries' % len(X_test))
    
    X_train, X_test = dataset_transform(D)
    
    t0 = time.time()
    memory_usage_before = psutil.Process().memory_info().rss / 1024
    algo.fit(X_train)
    build_time = time.time() - t0
    index_size = psutil.Process().memory_info().rss / 1024 - memory_usage_before
    print('Built index in', build_time)
    print('Index size: ', index_size)
    
    
        
        
    best_search_time = float('inf')

    for i in range(5):
        print('Run %d/5...' % (i + 1))
        n_items = 0
        results = []


        for x in X_test:
            start = time.time()
            candidates = algo.query(x, k)
            search_time = (time.time() - start)
            candidates = [(int(idx), float(metrics[distance](x, X_train[idx]))) \
                      for idx in candidates]#other distances
            n_items += 1

            if n_items % 1000 == 0:
                print("Processed %d/%d queries..." % (n_items, len(X_test)))

            results.append((search_time, candidates))


        total_search_time = sum(time for time, _ in results)
        total_candidates = sum(len(candidates) for _, candidates in results)
        search_time = total_search_time / len(X_test)
        avg_candidates = total_candidates / len(X_test)#和k有什么区别
        best_search_time = min(best_search_time, search_time)

    attrs = {
    "best_search_time": best_search_time,
    "candidates": avg_candidates,
    "name": str(algo),
    "count": int(k),
    "build_time": build_time,
    "index_size": index_size,
    "dataset": dataset,
    "algo": algo_name,
    "building_arguments": re.sub(r'\W+', '_', json.dumps(building_arguments, sort_keys=True)).strip('_'),
    "query_argument": '1'
    }

    store_results(dataset, k, attrs['query_argument'], attrs, results, attrs['building_arguments'])

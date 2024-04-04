import h5py
import json
import os
import re
import traceback


def get_result_filename(dataset=None, count=None, 
                        query_arguments=None, algo=None, building_arguments=None):
    d = ['results']
    if dataset:
        d.append(dataset)
    if count:
        d.append(str(count))
    
    if building_arguments:
        d.append(algo)
        
        #\W+ means non alphabet 
        d.append(building_arguments) 
        
        d.append(query_arguments + ".hdf5" )#change the definition into string then divide them by _
    return os.path.join(*d)


def store_results(dataset, count, query_arguments, attrs, results, building_arguments):
    fn = get_result_filename(
        dataset, count, query_arguments,  attrs['algo'], building_arguments)
    head, tail = os.path.split(fn)
    if not os.path.isdir(head):
        os.makedirs(head)
    f = h5py.File(fn, 'w')
    for k, v in attrs.items():
        f.attrs[k] = v
    times = f.create_dataset('times', (len(results),), 'f')
    neighbors = f.create_dataset('neighbors', (len(results), count), 'i')
    distances = f.create_dataset('distances', (len(results), count), 'f')
    for i, (time, ds) in enumerate(results):
        times[i] = time
        neighbors[i] = [n for n, d in ds] + [-1] * (count - len(ds))
        distances[i] = [d for n, d in ds] + [float('inf')] * (count - len(ds))
    f.close()

    
def load_all_results(dataset=None, count=None):
    for root, _, files in os.walk(get_result_filename(dataset, count)):
        for fn in files:
            if os.path.splitext(fn)[-1] != '.hdf5':
                continue
            try:
                f = h5py.File(os.path.join(root, fn), 'r+')
                properties = dict(f.attrs)
                building_arguments = root.split('/')[-1]
                yield properties, f, fn, building_arguments
                f.close()
            except:
                print('Was unable to read', fn)
                traceback.print_exc()

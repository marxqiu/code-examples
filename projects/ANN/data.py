import h5py
import numpy as np
import os
import random

from urllib.request import urlopen
from urllib.request import urlretrieve


import progressbar


pbar = None


def show_progress(block_num, block_size, total_size):
    global pbar
    if pbar is None:
        pbar = progressbar.ProgressBar(maxval=total_size)
        pbar.start()

    downloaded = block_num * block_size
    if downloaded < total_size:
        pbar.update(downloaded)
    else:
        pbar.finish()
        pbar = None




def download(src, dst):
    if not os.path.exists(dst):
        # TODO: should be atomic
        print('downloading %s -> %s...' % (src, dst))
        urlretrieve(src, dst, show_progress)


def get_dataset_fn(dataset):
    if not os.path.exists('data'):
        os.mkdir('data')
    return os.path.join('data', '%s.hdf5' % dataset)


def get_dataset(which):
    hdf5_fn = get_dataset_fn(which)
    try:
        url = 'http://ann-benchmarks.com/%s.hdf5' % which
        download(url, hdf5_fn)
    except:
        print("Cannot download %s" % url)
        if which in DATASETS:
            print("Creating dataset locally")
            DATASETS[which](hdf5_fn)
    hdf5_f = h5py.File(hdf5_fn, 'r')

    # here for backward compatibility, to ensure old datasets can still be used with newer versions
    # cast to integer because the json parser (later on) cannot interpret numpy integers
    dimension = int(hdf5_f.attrs['dimension']) if 'dimension' in hdf5_f.attrs else len(hdf5_f['train'][0])

    return hdf5_f, dimension


def sparse_to_lists(data, lengths):
    X = []
    index = 0
    for l in lengths:
        X.append(data[index:index+l])
        index += l

    return X

def dataset_transform(dataset):
    if dataset.attrs.get('type', 'dense') != 'sparse':
        return np.array(dataset['train']).astype(np.float32), np.array(dataset['test']).astype(np.float32)

    # we store the dataset as a list of integers, accompanied by a list of lengths in hdf5
    # so we transform it back to the format expected by the algorithms here (array of array of ints)
    return sparse_to_lists(dataset['train'], dataset['size_train']), sparse_to_lists(dataset['test'], dataset['size_test'])


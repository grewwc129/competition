import pandas as pd
from .config import config
from .general_utils import drop_non_numeric, encode_name
from contextlib import contextmanager
import os
from typing import Callable, List
import threading
import gc
import h5py
import numpy as np


root = "D:/data/"


@contextmanager
def _log(func: Callable, count: int):
    print("entering: {}".format(func.__name__), count)
    try:
        yield
    finally:
        print("leaving: {}".format(func.__name__), count)



def write_h5(batch_size: int=10000):
    """write a giant csv file into many hdf5 files
    """
    count = 0
    df = pd.read_csv(config.train_file_name, iterator=True, low_memory=False)
    while True:
        try:
            cur_df = df.get_chunk(batch_size)
            with _log(_write_single, count):
                _write_single(
                    cur_df, filename=os.path.join(root, f"{count}.h5"))
            count += 1
        except StopIteration:
            break


def _write_single(df, filename: str, preprocess: bool=True):
    if preprocess:
        df.drop('id', axis=1, inplace=True)
        df = drop_non_numeric(df)
        df.answer = df.answer.apply(encode_name)
        df = df.astype('float32')
        df = df.values

    dirname = os.path.dirname(filename)
    with h5py.File(filename, mode='w') as f:
        f.create_dataset('data', data=df)


def _merge_two_h5(f1: str, f2: str):
    def get_new_name(name1: str, name2: str):
        dirname = os.path.dirname(name1)
        name1, name2 = list(map(os.path.basename, (name1, name2)))
        name1, name2 = (name.split('.')[0] for name in (name1, name2))
        return os.path.join(dirname, f"{name1}#{name2}.h5")

    arrays = []
    threads = []
    def func(filename: str):
        data = read_h5(filename)
        arrays.append(data)

    threads.append(threading.Thread(target=func, args=(f2, )))
    threads.append(threading.Thread(target=func, args=(f1, )))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    merged = np.r_[arrays[0], arrays[1]]

    del arrays
    gc.collect()

    merged_name = get_new_name(f1, f2)
    _write_single(merged, merged_name, preprocess=False)

    os.remove(f1)
    os.remove(f2)


def merge_h5():
    """call this function after "write_h5"
    """
    filenames = [os.path.join(root, filename) for filename in os.listdir(root)]
    filenames = [filename for filename in filenames if os.path.isfile(filename)]
    if len(filenames) <= 1:
        return

    for i in range(0, len(filenames)-1, 2):
        _merge_two_h5(filenames[i], filenames[i+1])

    merge_h5()


def read_h5(filename: str)->np.ndarray:
    with h5py.File(filename, 'r') as f:
        return np.array(f.get('data'))

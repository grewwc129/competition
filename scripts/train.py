import pandas as pd
from .data_utils import trim_df

def train():
    batch_size = 1024
    for i in range(10):
        print("#"*30)
        print('iteration: {}'.format(i))
        df = pd.read_csv(fname, iterator=True)
        count = 0
        while 1:
            try:
                cur = df.get_chunk(batch_size)
                train_x, train_y = trim_df(cur)
                print("*" * 30)
                print("iteration: {}, count: {}".format(i, count + 1))
                count += 1
                model.fit(train_x, train_y, epochs=20, validation_split=0.05)
                print("*" * 30)
            except StopIteration:
                break


def train_binary():
    batch_size = 1024
    for i in range(10):
        print("#"*30)
        print('iteration: {}'.format(i))
        df = pd.read_csv(fname, iterator=True)
        count = 0
        while 1:
            try:
                cur = df.get_chunk(batch_size)
                train_x, train_y = trim_df(cur)
                print("*" * 30)
                print("iteration: {}, count: {}".format(i, count + 1))
                count += 1
                model.fit(train_x, train_y, epochs=20, validation_split=0.05)
                print("*" * 30)
            except StopIteration:
                break

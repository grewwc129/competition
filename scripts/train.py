import pandas as pd
from .data_utils import *
from .nn_resnet import resnet
from .general_utils import decay_lr


def train():
    batch_size = 1024 * 8
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
                model.fit(train_x, train_y, epochs=5, validation_split=0.05)
                print("*" * 30)
            except StopIteration:
                break


def train_binary(target_name):
    batch_size = 1024*32
    model = resnet(2, input_shape=(config.num_bins, 1))
    fname = "update_new_columns_trains_sets.csv"
    for i in range(5):
        print("#"*30)
        print('iteration: {}'.format(i))
        df = pd.read_csv(fname, iterator=True, low_memory=False)
        count = 0
        if count > 0:
            decay_lr(model, 0.7)
        while 1:
            try:
                cur = df.get_chunk(batch_size)
                train_x, train_y = trim_df_binary(cur, target_name)
                print("*" * 30)
                print("iteration: {}, count: {}".format(i, count + 1))
                count += 1
                model.fit(train_x, train_y, epochs=10)
                print("*" * 30)
            except StopIteration:
                break
        model.save("./temp_{}.h5".format(i), overwrite=True)


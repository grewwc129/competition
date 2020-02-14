import pandas as pd


training = "update_new_columns_trains_sets.csv"
val = "val_sets_v1.csv"

count = {'star': 0, 'galaxy': 0, 'qso': 0}


def main(fname):
    df = pd.read_csv(fname, iterator=True, low_memory=0)
    it = 0
    while 1:
        try:
            cur = df.get_chunk(1024)
            print(it)
            it += 1
            answer = cur['answer']
            count['star'] += sum(answer == 'star')
            count['galaxy'] += sum(answer == 'galaxy')
            count['qso'] += sum(answer == 'qso')
            rest = cur[(answer != 'star') & (answer != 'galaxy') & (answer != 'qso')]

            if len(rest) != 0:
                print(rest)

            print(count)
        except StopIteration:
            break


main(training)

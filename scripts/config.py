class Config:
    def __init__(self):
        # self.train_file_name = "update_new_columns_trains_sets.csv"
        self.train_file_name = "train_data.h5"
        self.class_label = {'star': 0, 'galaxy': 1, 'qso': 2}
        self.num_bins = 400


config = Config()

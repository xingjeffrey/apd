## copied from https://stackoverflow.com/a/71592809
from tensorflow.keras.utils import Sequence
import numpy as np   

class DataGenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(self.x[0].shape[0] / float(self.batch_size))) ## go into the first slice, and index axis 0

    def getx(self, x, idx):
        return x[idx * self.batch_size : (idx + 1) * self.batch_size]
    
    def __getitem__(self, idx):
        batch_x = [
            self.getx(x, idx)
            for x in self.x
        ]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
        yield batch_x, batch_y
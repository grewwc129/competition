from keras.models import * 
from keras.layers import * 
import keras.backend as K 


def dense(input_shape=(2600, 1)):
    x = Input(shape=input_shape)
    
    
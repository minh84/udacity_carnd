from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda

def create_nvidia_model():
    model = Sequential()
    model.add(Lambda(lambda x: (x/255.0) - 0.5),
                     input_shape = (66,200, 3))
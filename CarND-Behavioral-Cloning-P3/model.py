from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D, Dropout

def build_nvidia(input_shape = (160,320,3), cropping=((50, 20), (0, 0))):
    model = Sequential()

    # cropping
    model.add(Cropping2D(cropping=cropping, input_shape=input_shape))

    # simple normalization
    model.add(Lambda(lambda x: x / 255.0 - 0.5))

    # first 3 conv-layers use kernel (5, 5) with stride (2, 2)
    # filters-depths are [24, 36, 48]
    filter_depths = [24, 36, 48]
    for d in filter_depths:
        model.add(Conv2D(d, 5, strides=(2, 2), activation='relu'))

    # the last 2 conv-layers use kernel (3, 3) with stride (1, 1)
    filter_depths = [64, 64]
    for d in filter_depths:
        model.add(Conv2D(d, 3, strides=(1, 1), activation='relu'))

    # flatten
    model.add(Flatten())

    # fully-connected layers
    hidden_dims = [100, 50, 10]
    for h in hidden_dims:
        model.add(Dense(h, activation='relu'))
        model.add(Dropout(0.5))

    # output-regressor
    model.add(Dense(1))

    return model


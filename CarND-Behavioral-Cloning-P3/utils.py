import cv2
import numpy as np
import matplotlib.image as mpimg
import pandas
import os

# parameter to control how we crop the image
CROP_TOP     = 60
CROP_BOTTOM  = 25

# parameter to control how we resize the image
INPUT_WIDTH  = 200
INPUT_HEIGHT = 66

# input shape for model
INPUT_SHAPE = (INPUT_HEIGHT, INPUT_WIDTH, 3)

def crop(img, top, bottom):
    return img[top:-bottom, :, :]


def resize(img, width, height):
    return cv2.resize(img, (width, height), cv2.INTER_AREA)


def preprocess(img,
               crop_top=CROP_TOP,
               crop_bottom=CROP_BOTTOM,
               resize_width=INPUT_WIDTH,
               resize_height=INPUT_HEIGHT):
    # crop img
    img = crop(img, crop_top, crop_bottom)

    # resize img
    img = resize(img, resize_width, resize_height)

    return img


def multiple_camera(left_center_right, steering, correction):
    lcr = np.random.choice(3)
    if lcr == 0:
        steering += correction
    elif lcr == 2:
        steering -= correction

    return left_center_right[lcr], steering


def random_flip(img, steering):
    if np.random.randint(2) == 0:
        return img, steering
    else:
        return np.fliplr(img), -steering

def random_shadow(img):
    H, W, _ = img.shape

    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = 0, W * np.random.rand()
    x2, y2 = H, W * np.random.rand()

    xm, ym = np.mgrid[0:H, 0:W]

    # mask above the line to be 1
    mask = np.zeros((H, W))
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust Light
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # convert to HLS(Hue, Light, Saturation) then adjust Light
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(img):
    # convert to HSV (Hue, Saturation, Value) where Value is for brightness.
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    # randomly adjust brightness
    ratio = 0.5 + np.random.uniform()
    hsv[:, :, 2] = ratio * hsv[:, :, 2]
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255

    # convert back to RGB
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def random_translate(img, steering, range_x, range_y):
    # see example from
    #    http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html
    trans_x = range_x * (np.random.uniform() - 0.5)
    trans_y = range_y * (np.random.uniform() - 0.5)

    steering += trans_x * 0.002
    trans_matrix = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    H, W, _ = img.shape

    return cv2.warpAffine(img, trans_matrix, (W, H)), steering

def get_jittered_data(img, steering):
    img = mpimg.imread(img)

    # random flip
    img, steering = random_flip(img, steering)

    # random shadow
    img = random_shadow(img)

    # random brightness
    img = random_brightness(img)

    # random translate
    img, steering = random_translate(img, steering, 80, 10)

    return img, steering

def get_next_train_data(dataset, row_idx, use_multiple_camera, correction):
    '''
    
    :param dataset: 
    :param row_idx: 
    :param use_multiple_camera: 
    :param correction: 
    :return: 
    '''
    if use_multiple_camera:
        return multiple_camera(dataset[['left', 'center', 'right']].values[row_idx],
                               dataset[['steering']].values[row_idx],
                               correction)
    else:
        return dataset[['center', 'steering']].values[row_idx]


def train_generator(dataset, input_size, batch_size, correction, use_multiple_camera=True):
    num_samples = dataset.shape[0]
    inputs = np.zeros((batch_size, *input_size), dtype=np.uint8)
    targets = np.zeros((batch_size), dtype=np.float32)

    while 1:

        for i in range(batch_size):
            row_idx = np.random.randint(num_samples)
            img, steering = get_next_train_data(dataset,
                                                row_idx,
                                                use_multiple_camera,
                                                correction)

            img, targets[i] = get_jittered_data(img, steering)
            inputs[i] = preprocess(img)

        yield inputs, targets

def valid_generator(dataset, input_size, batch_size):
    num_samples = dataset.shape[0]

    inputs = np.zeros((batch_size, *input_size), dtype=np.uint8)
    targets = np.zeros((batch_size), dtype=np.float32)

    steps_per_epoch = num_samples // batch_size

    while 1:
        for i in range(steps_per_epoch):
            offest = i * batch_size
            for j in range(batch_size):
                img, steering = mpimg.imread(dataset['center'][offest + j]), dataset['steering'][offest + j]
                inputs[j] = preprocess(img)
                targets[j] = steering

            yield inputs, targets


class DataConverter:
    '''
    convert relative path to full-path
    '''
    def __init__(self, data_dir):
        self._data_dir = data_dir

    def __call__(self, text):
        return os.path.join(self._data_dir, text.strip())

def load_data(data_dir):
    '''
    load data from directory, we support the following two cases
    1) data directory contains `dataset.csv`, then we load the `dataset.csv`
    2) data directory contains few sub-folder each contains driving_log.csv & IMG for example
        data/
            drive/
                driving_log.csv
                IMG
            recovery/
                driving_log.csv
                IMG
            udacity/
                driving_log.csv
                IMG
       we expect each driving_log.csv contains the header row:
            center,left,right,steering,throttle,brake,speed
        
    :param data_dir: 
    :return: pandas.DataFrame
    '''
    if not os.path.isdir(data_dir):
        raise Exception('Data dir {} does NOT exist'.format(data_dir))

    dataset_file = os.path.join(data_dir, 'dataset.csv')
    if os.path.isfile(os.path.join(dataset_file)):
        return pandas.read_csv(dataset_file)
    else:
        list_dirs = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
        sub_dirs = [x for x in list_dirs if os.path.isdir(x)]

        datas = []
        for dir in sub_dirs:
            dc = DataConverter(dc)
            data = pandas.read_csv(os.path.join(dir, 'driving_log.csv'),
                                   converters={'left': dc, 'right': dc, 'center': dc})
            datas.append(data)

        # concat all data
        dataset =  pandas.concat(datas, index=False)
        # save to csv so next time we don't need to re-do this
        dataset.to_csv(dataset_file)

        return dataset

def split_data(dataset, split_frac):
    '''
    Split dataset to train/validation
    :param dataset: input pandas.DataFrame
    :param split_frac: split ratio
    :return: train, valid
    '''
    train, valid = np.split(dataset.sample(frac=1, random_state=45), [int(split_frac * len(dataset))])

    train = pandas.DataFrame(train.values, columns=dataset.columns)
    valid = pandas.DataFrame(valid.values, columns=dataset.columns)

    print('\n-------------------------------------------------')
    print('training-samples   {}'.format(train.shape[0]))
    print('validation-samples {}'.format(valid.shape[0]))
    print('-------------------------------------------------\n')

    return train, valid

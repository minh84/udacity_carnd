import numpy as np

class BaseTransform(object):
    '''
    A base class that used to pre-processing data
        e.g: convert RGB to Gray-scale
                     RGB to YUV e.t.c
    '''
    def __init__(self, transform_func = None):
        self._transform_func = transform_func

    def fit(self, training_inputs, training_labels = None):
        '''
        Some tranformer need to fit to training inputs for example
            to compute mean/variance of training inputs
            so later can applied transform to validation/test inputs
        :param training_inputs: training inputs is a nd-array
        :param training_labels: training labels is optional
        :return: self
        '''
        return self

    def transform(self, inputs, labels = None):
        '''
        Run transform on inputs
        :param inputs: a nd-array with shape [N,...] where N is number of samples
        :param labels: a nd-array of corresponding labels
        :return: a nd-array after the transform applied
        '''
        if self._transform_func is None:
            raise Exception('Can NOT run transform since self._transform_func')

        return np.array([self._transform_func(x) for x in inputs]), labels

    def fit_and_transform(self, inputs, labels = None):
        return self.fit(inputs, labels).transform(inputs, labels)

class StandardTransform(BaseTransform):
    def __init__(self):
        super(StandardTransform, self).__init__(None)

    def fit(self, training_inputs, training_labels = None):
        self._mean = np.mean(training_inputs, axis=0, keepdims=True)
        self._stddev = np.std(training_inputs, axis=0, keepdims=True)

        return self

    def transform(self, inputs, labels = None):
        return (inputs - self._mean) / self._stddev, labels

def label_to_indices(labels):
    '''
    we build a map label -> list of indices that have the corresponding label
    :param labels: a list of labels each is an integer each in range [0, nb_classes) 
    :return: 
            nb_classes: a number of classes
            label_to_idx: a map label -> list of indices 
    '''
    label_to_idx = {}
    nb_classes = np.max(labels) + 1

    for i in range(nb_classes):
        label_to_idx[i] = np.where(labels == i)[0]
    return nb_classes, label_to_idx

AUG_DEFAULT = {'jitter'         : 4,
               'jitter_balance' : 3000}

class AugmentTransform(BaseTransform):
    def __init__(self, jitter_func, **augment_settings):
        super(AugmentTransform, self).__init__(jitter_func)
        self._augtype = augment_settings.get('augtype', 'jitter')
        assert self._augtype in ['jitter', 'jitter_balance'], 'Only support augtype = jitter or jitter_balance'

        self._auglevel     = augment_settings.get('auglevel', AUG_DEFAULT[self._augtype])
        self._use_original = augment_settings.get('use_original', True)

    def transform(self, inputs, labels = None):
        assert labels is not None, 'AugmentTransform.transform requires inputs and labels'

        nb_classes, labels_map = label_to_indices(labels)

        aug_inputs = []
        aug_labels = []

        # this creates an additional self._auglevel x N samples (each additional image is a jittered of original input)
        if self._augtype == 'jitter':
            for i in range(nb_classes):
                label = i

                for j in labels_map[i]:
                    # original input
                    input = inputs[j]
                    if self._use_original:
                        aug_inputs.append(input)
                        aug_labels.append(label)

                    # jittered input
                    for k in range(self._auglevel):
                        aug_inputs.append(self._transform_func(input))
                        aug_labels.append(label)

        elif self._augtype == 'jitter_balance':
            for i in range(nb_classes):
                nb_aug_input = self._auglevel
                label = i
                indices = labels_map[i]
                num_original_inputs = len(indices)
                if num_original_inputs == 0:   #
                    print('WARNING label {} has NO input'.format(i))
                    continue

                if self._use_original:
                    aug_inputs += [input for input in inputs[indices]]
                    aug_labels += [label] * num_original_inputs
                    nb_aug_input -= num_original_inputs

                for j in range(nb_aug_input):
                    idx = indices[j % num_original_inputs]
                    aug_inputs.append(self._transform_func(inputs[idx]))
                    aug_labels += [label]
        else:
            raise Exception('Unknown augmentation-type {}'.format(self._augtype))

        return np.array(aug_inputs), np.array(aug_labels)

class TransformPipeline(object):
    '''
    This class is used to apply a list of transform or data
    '''
    def __init__(self, trans_list):
        '''
        Constructor of TransformPipeline
        :param trans_list: a list of transform object
        '''
        self._trans_list = trans_list

    def fit_and_transform(self, inputs, labels = None):

        for trans in self._trans_list:
            inputs, labels = trans.fit_and_transform(inputs, labels)

        return inputs, labels

    def transform(self, inputs, labels = None):
        for trans in self._trans_list:
            inputs, labels = trans.transform(inputs, labels)
        return inputs, labels

class Minibatch(object):
    def __init__(self, inputs, labels):
        self._inputs = inputs
        self._labels = labels

    def input_shape(self):
        return self._inputs.shape[1:]

    def get_batches(self, batch_size, shuffle = False, rounded = False):

        N = self._inputs.shape[0]
        idx = np.arange(N)
        if shuffle:
            np.random.shuffle(idx)

        if rounded:
            num_batches = N // batch_size
        else:
            num_batches = (N + batch_size - 1) // batch_size

        for i in range(num_batches):
            idxi = idx[i*batch_size : (i + 1)*batch_size]
            yield self._inputs[idxi], self._labels[idxi]

    @property
    def inputs(self):
        return self._inputs

    @property
    def labels(self):
        return self._labels

class Dataset(object):
    def __init__(self, trans_pipeline, training_set, validation_set, test_set, jitter_func = None):
        self._trans_pipeline = trans_pipeline

        self._train_set = Minibatch(*self._trans_pipeline.fit_and_transform(*training_set))
        self._val_set   = Minibatch(*self._trans_pipeline.transform(*validation_set))
        self._test_set  = Minibatch(*self._trans_pipeline.transform(*test_set))

        self._jitter_trans = None
        if jitter_func is not None:
            self._jitter_trans = BaseTransform(jitter_func)
            self._raw_training_set = training_set

    def input_shape(self):
        return self._train_set.input_shape()

    def transform(self, inputs, labels = None):
        return self._trans_pipeline.transform(inputs, labels)

    def get_training_batches(self, batch_size):
        return self._train_set.get_batches(batch_size, shuffle=True, rounded=True)

    def jitter_training(self):
        if self._jitter_trans is not None:
            # we create jittered data from orginal dataset
            jitter_training_set = self._jitter_trans.transform(*self._raw_training_set)
            self._train_set = Minibatch(*self._trans_pipeline.transform(*jitter_training_set))

    @property
    def train_set(self):
        return self._train_set

    @property
    def val_set(self):
        return self._val_set

    @property
    def test_set(self):
        return self._test_set
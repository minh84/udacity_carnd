import numpy as np
import os
import cv2
import pickle

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

    def tranform(self, inputs, labels = None):
        '''
        Run transform on inputs
        :param inputs: a nd-array with shape [N,...] where N is number of samples
        :param inputs: a nd-array with shape [N,...] where N is number of samples
        :return: a nd-array aftered applied the transform
        '''
        if self._transform_func is None:
            raise Exception('Can NOT run transform since self._transform_func')

        return np.array([self._transform_func(x) for x in inputs])


    def transform_and_pickle(self, inputs, out_filename):
        '''
        Since tranform can take a lot of time, this method run transform on inputs 
        and then pickle result to a file 
        :param inputs: a nd-array
        :return: a file name
        '''
        output = self.tranform(inputs)

        out_dir = os.path.dirname(out_filename)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)

        with open(out_filename, 'wb') as f:
            pickle.dump(output, f)

class NormalizeTransform(BaseTransform):
    def __init__(self):
        super(NormalizeTransform, self).__init__()

    def fit(self, training_inputs):
        self._mean = np.mean(training_inputs, axis=0, keepdims=True)
        self._stddev = np.sqrt(np.var(training_inputs, axis=0, keepdims=True))

    def tranform(self, inputs):
        return (inputs - self._mean) / self._stddev
import collections

import tensorflow as tf
from tensorflow.contrib.layers import flatten

from .session import Session

class NeuralNetwork(object):
    '''
    A thin wrapper of Tensorflow api
    Allow similar keras model construction e.g
        NeuralNetwork(graph)
            .input(input_shape)
            .conv2d([kernel_height, kernel_width], out_channels)            
            
    Note that it doesn't check the order of construction 
    so it's user's responsability to give a sensible construction i.e
            first: should be inputs, then follow by conv2d/dense e.t.c            
    '''
    def __init__(self):
        # keep a graph to itself
        self.graph = tf.Graph()

        # keep a map of intermediate layers
        self.last_layer = None
        self.layers = {}
        self.layers_count = collections.defaultdict(int)

    def input(self, input_shape):
        '''
        add input layers:
            self.inputs: contain inputs
            self.labels: contain labels
            
        :param input_shape: a list/tuple 
        :return: self
        '''
        with self.graph.as_default():
            with tf.variable_scope('data'):
                self.inputs = tf.placeholder(tf.float32, [None, *input_shape], name = 'inputs')
                self.labels = tf.placeholder(tf.int64, [None], name = 'labels')
                self.is_training = tf.placeholder(tf.bool, name = 'is_training')


            self.last_layer = self.inputs

        return self

    def conv(self, ksize, out_channels, strides = (1,1), padding = 'SAME', activation = tf.nn.relu):
        '''
        add a convulation layer
        :param ksize: tuple/list of [H,W] for kernel-size
        :param out_channels: number of output filters
        :param strides: control stride in conv2d
        :param padding: control padding default 'valid'
        :param activation: activation default 'relu'
        :return: self
        '''
        with self.graph.as_default():
            # update counter
            self.layers_count['conv'] += 1
            layer_name = 'conv_{}'.format(self.layers_count['conv'])

            # we use xavier init
            xavier_init = tf.contrib.layers.xavier_initializer()

            # get input depth
            in_channels = self.last_layer.get_shape().as_list()[-1]

            with tf.variable_scope(layer_name):
                W = tf.get_variable('weights', shape=[*ksize, in_channels, out_channels], initializer=xavier_init)
                b = tf.get_variable('biases',  shape=[out_channels], initializer=tf.zeros_initializer())
                conv = tf.nn.conv2d(self.last_layer, W, b,
                                    strides = [1, *strides, 1],
                                    padding=padding)

                if activation is not None:
                    conv = activation(conv)

            self.layers[layer_name] = conv

            self.last_layer = conv

        return self

    def dropout(self, keep_prob):
        with self.graph.as_default():
            # update counter
            self.layers_count['dropout'] += 1
            layer_name = 'dropout_{}'.format(self.layers_count['dropout'])

            self.last_layer = tf.cond(self.is_training,
                                      lambda : tf.nn.dropout(self.last_layer, keep_prob),
                                      lambda : self.last_layer)

            self.layers[layer_name] = self.last_layer

        return self

    def max_pool(self, ksize, stride, padding = 'SAME'):
        with self.graph.as_default():
            self.layers_count['max_pool'] += 1
            layer_name = 'max_pool_{}'.format(self.layers_count['max_pool'])

            self.last_layer = tf.nn.max_pool(self.last_layer,
                                             ksize=[1,ksize, ksize, 1],
                                             strides=[1,stride, stride, 1],
                                             padding=padding)
            self.layers[layer_name] = self.last_layer

        return self

    def batch_norm(self, decay = 0.9, epsilon = 1e-6):
        with self.graph.as_default():
            # update counter
            self.layers_count['batch_norm'] += 1
            layer_name = 'batch_norm_{}'.format(self.layers_count['batch_norm'])

            self.last_layer = tf.contrib.layers.batch_norm(self.last_layer,
                                                           center  = True,
                                                           scale   = True,
                                                           decay   = decay,
                                                           epsilon = epsilon,
                                                           updates_collections = None,
                                                           is_training         = self.is_training)
            self.layers[layer_name] = self.last_layer

        return self

    def multi_scale(self, layer_ids, max_pools):
        '''
        Create a multi-scale layers
            each is a concatenate of few other layers
        :param   layers:    a list of layer names
        :param   max_pools: a list of max_pool settings (ksize, stride)
        :return: self
        '''
        with self.graph.as_default():
            # update counter
            self.layers_count['multi_scale'] += 1
            layer_name = 'multi_scale_{}'.format(self.layers_count['multi_scale'])

            multi_scale = []
            for layer_id, max_pool in zip(layer_ids, max_pools):
                layer = self.layers[layer_id]
                if max_pool is not None:
                    ksize, stride = max_pool
                    layer = tf.nn.max_pool(layer, ksize=[1,ksize, ksize, 1],
                                           strides=[1,stride, stride, 1], padding='SAME')
                multi_scale.append(flatten(layer))

            self.last_layer = tf.concat(multi_scale, axis=1)

            self.layers[layer_name] = self.last_layer
        return self

    def dense(self, out_channels, activation = tf.nn.relu):
        with self.graph.as_default():
            # update counter
            self.layers_count['dense'] += 1
            layer_name = 'dense_{}'.format(self.layers_count['dense'])

            # we use xavier init
            xavier_init = tf.contrib.layers.xavier_initializer()

            # get input depth
            in_channels = self.last_layer.get_shape().as_list()[-1]

            with tf.variable_scope(layer_name):
                W = tf.get_variable('weights', shape=[in_channels, out_channels], initializer=xavier_init)
                b = tf.get_variable('biases',  shape=[out_channels], initializer=tf.zeros_initializer())

                fc = tf.matmul(self.last_layer, W) + b

                if activation is not None:
                    fc = activation(fc)

            self.layers[layer_name] = fc

            self.last_layer = fc

class NeuralNetworkClassifier(object):
    def __init__(self, network, top_k = 5):
        # get hold of graph, inputs, labels & last_layer (which is logits)
        self.graph  = network.graph
        self.inputs = network.inputs
        self.labels = network.labels
        self.logits = network.last_layer

        # we also need this to do train/predict
        self.is_training = network.is_training

        # add loss-op
        self.add_loss_op()

        # add prediction-op
        self.add_pred_op(top_k = top_k)

    def add_loss_op(self):
        # construct loss & cost (with softmax-cross-entropy)
        with self.graph.as_default():
            self.loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                                         labels=self.labels))

    def add_prediction_op(self, top_k):
        with self.graph.as_default():
            # predition is simply max-activation
            self.pred_op = tf.arg_max(self.logits, axis = 1)

            # probability is obtained via softmax
            self.prob = tf.nn.softmax(self.logits)

            # compute top_k softmax
            self.top_k = tf.nn.top_k(self.prob, top_k)

    def predict(self, session, inputs):
        return session.run(self.pred_op, feed_dict = {self.inputs : inputs, self.is_training : False})

    def get_top_prob(self, session, inputs):
        return session.run(self.top_k, feed_dict = {self.inputs : inputs, self.is_training : False})
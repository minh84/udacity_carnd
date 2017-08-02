import collections
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import flatten

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
        self._graph = tf.Graph()

        # keep a map of intermediate layers
        self._last_layer = None
        self._layers = {}
        self._layers_count = collections.defaultdict(int)

        # net type is either classifier or regressor
        self._net_type = None

        # we always need a loss function
        self._loss_op = None

    def input(self, input_shape):
        '''
        add input layers:
            self._inputs: contain inputs
            self._labels: contain labels
            
        :param input_shape: a list/tuple 
        :return: self
        '''
        with self._graph.as_default():
            with tf.variable_scope('data'):
                self._inputs = tf.placeholder(tf.float32, [None, *input_shape], name = 'inputs')
                self._labels = tf.placeholder(tf.int64, [None], name = 'labels')
                self._is_training = tf.placeholder(tf.bool, name = 'is_training')


            self._last_layer = self._inputs

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
        with self._graph.as_default():
            # update counter
            self._layers_count['conv'] += 1
            layer_name = 'conv_{}'.format(self._layers_count['conv'])

            # get input depth
            in_channels = self._last_layer.get_shape().as_list()[-1]

            # we use either xavier init or trunctated normal
            initializer = tf.contrib.layers.xavier_initializer()

            with tf.variable_scope(layer_name):
                W = tf.get_variable('weights', shape=[*ksize, in_channels, out_channels], initializer=initializer)
                b = tf.get_variable('biases',  shape=[out_channels], initializer=tf.zeros_initializer())
                conv = tf.nn.conv2d(self._last_layer, W, strides = [1, *strides, 1],
                                    padding=padding) + b

                if activation is not None:
                    conv = activation(conv)

            self._layers[layer_name] = conv

            self._last_layer = conv

        return self

    def dropout(self, keep_prob):
        with self._graph.as_default():
            # update counter
            self._layers_count['dropout'] += 1
            layer_name = 'dropout_{}'.format(self._layers_count['dropout'])

            self._last_layer = tf.cond(self._is_training,
                                      lambda : tf.nn.dropout(self._last_layer, keep_prob),
                                      lambda : self._last_layer)

            self._layers[layer_name] = self._last_layer

        return self

    def max_pool(self, ksize, stride, padding = 'SAME'):
        with self._graph.as_default():
            self._layers_count['max_pool'] += 1
            layer_name = 'max_pool_{}'.format(self._layers_count['max_pool'])

            self._last_layer = tf.nn.max_pool(self._last_layer,
                                             ksize=[1,ksize, ksize, 1],
                                             strides=[1,stride, stride, 1],
                                             padding=padding)
            self._layers[layer_name] = self._last_layer

        return self

    def batch_norm(self, decay = 0.9, epsilon = 1e-6):
        with self._graph.as_default():
            # update counter
            self._layers_count['batch_norm'] += 1
            layer_name = 'batch_norm_{}'.format(self._layers_count['batch_norm'])

            self._last_layer = tf.contrib.layers.batch_norm(self._last_layer,
                                                           center  = True,
                                                           scale   = True,
                                                           decay   = decay,
                                                           epsilon = epsilon,
                                                           updates_collections = None,
                                                           is_training         = self._is_training)
            self._layers[layer_name] = self._last_layer

        return self

    def multi_scale(self, layer_ids, max_pools):
        '''
        Create a multi-scale layers
            each is a concatenate of few other layers
        :param   layers:    a list of layer names
        :param   max_pools: a list of max_pool settings (ksize, stride)
        :return: self
        '''
        with self._graph.as_default():
            # update counter
            self._layers_count['multi_scale'] += 1
            layer_name = 'multi_scale_{}'.format(self._layers_count['multi_scale'])

            multi_scale = []
            for layer_id, max_pool in zip(layer_ids, max_pools):
                layer = self._layers[layer_id]
                if max_pool is not None:
                    ksize, stride = max_pool
                    layer = tf.nn.max_pool(layer, ksize=[1,ksize, ksize, 1],
                                           strides=[1,stride, stride, 1], padding='SAME')
                multi_scale.append(flatten(layer))

            self._last_layer = tf.concat(multi_scale, axis=1)

            self._layers[layer_name] = self._last_layer
        return self

    def flatten(self):
        with self._graph.as_default():
            self._last_layer = flatten(self._last_layer)
        return self

    def dense(self, out_channels, activation = tf.nn.relu):
        with self._graph.as_default():
            # update counter
            self._layers_count['dense'] += 1
            layer_name = 'dense_{}'.format(self._layers_count['dense'])

            # get input depth
            in_channels = self._last_layer.get_shape().as_list()[-1]

            # we use either xavier init or trunctated normal
            initializer = tf.contrib.layers.xavier_initializer()

            with tf.variable_scope(layer_name):
                W = tf.get_variable('weights', shape=[in_channels, out_channels], initializer=initializer)
                b = tf.get_variable('biases',  shape=[out_channels], initializer=tf.zeros_initializer())

                fc = tf.matmul(self._last_layer, W) + b

                if activation is not None:
                    fc = activation(fc)

            self._layers[layer_name] = fc

            self._last_layer = fc
        return self

    def classifier(self, num_classes, top_k = 5, loss = tf.nn.sparse_softmax_cross_entropy_with_logits):
        with self._graph.as_default():
            # add logits layer
            self.dense(num_classes, activation=None)
            self._logits = self._last_layer

            # add loss
            self._loss_op = tf.reduce_mean(loss(logits=self._logits, labels=self._labels))

            # predition is simply max-activation
            self._pred_op = tf.argmax(self._logits, axis=1)

            # probability is obtained via softmax
            self._prob = tf.nn.softmax(self._logits)

            # compute top_k softmax
            self._top_k = tf.nn.top_k(self._prob, top_k)

            # set net type
            self._net_type = 'classifier'

        return self

    def get_feed_dict(self, inputs, labels=None, is_training=False):
        feed_dict = {self._inputs: inputs, self._is_training: is_training}
        if labels is not None:
            feed_dict[self._labels] = labels
        return feed_dict

    def predict(self, session, inputs):
        if self._net_type != 'classifier':
            raise Exception('Net is not a classifier, can NOT use self.predict')

        return session.run(self._pred_op, feed_dict = self.get_feed_dict(inputs))

    def top_k(self, session, inputs):
        if self._net_type != 'classifier':
            raise Exception('Net is not a classifier, can NOT use self.get_top_prob')

        return session.run(self._top_k, feed_dict = self.get_feed_dict(inputs))

    def get_layers(self, session, inputs, layer_ids):
        ops = []
        for layer_id in layer_ids:
            if layer_id not in self._layers:
                raise Exception('Layer {} is not in current net, it only has following layers\n\t{}'.format(layer_id,
                                                                                                            '\n\t'.join(self._layers.keys())))
            ops.append(self._layers[layer_id])

        return session.run(ops, feed_dict = self.get_feed_dict(inputs))

    @property
    def loss_op(self):
        return self._loss_op

    @property
    def graph(self):
        return self._graph
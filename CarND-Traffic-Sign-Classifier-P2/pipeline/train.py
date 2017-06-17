import numpy as np
import os
import tensorflow as tf
import tempfile

class TrainingSession(object):
    '''
    A thin wrapper class for tf.Session allows loading/saving more easily
    '''
    def __init__(self, net, max_to_keep=50, auto_save = True, check_point = None):
        self._net     = net
        self._graph   = self._net.graph
        self._loss_op = self._net.loss_op

        self._train_op = None

        self._auto_save = auto_save
        if self._auto_save:
            self._check_point = check_point
            if self._check_point is None:
                # create a temporary name to store this session file
                with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp:
                    self._check_point = temp.name

        with self._graph.as_default():
            self.sess  = tf.Session(graph=self._graph)
            self.saver = tf.train.Saver(max_to_keep=max_to_keep)

    def init_optimizer(self, learning_rate):
        with self._graph.as_default():
            self._train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self._loss_op)
            self.run(tf.global_variables_initializer())

    def train_and_evaluate_cost(self, inputs, labels):
        return self.run([self._train_op, self._loss_op],
                        feed_dict=self._net.get_feed_dict(inputs, labels, is_training=True))

    def predict(self, inputs):
        return self._net.predict(self, inputs)

    def top_k(self, inputs):
        return self._net.top_k(self, inputs)

    def get_layers(self, inputs, layer_ids):
        return self._net.get_layers(self, inputs, layer_ids)

    def score(self, dataset, batch_size):
        preds = []
        for inputs,_ in dataset.get_batches(batch_size):
            pred = self.predict(inputs)
            preds.extend(pred)

        return np.mean(np.array(preds) == dataset.labels)

    def load(self, checkpoint):
        '''
        Load the session state for checkpoint file
        :param checkpoint: a checkpoint file 
        :return: None
        '''
        self.saver.restore(self.sess, checkpoint)

    def save(self, checkpoint):
        '''
        Save the session state into a checkpoint
        :param checkpoint: 
        :return: None
        '''
        save_dir = os.path.dirname(checkpoint)
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)

        self.saver.save(self.sess, checkpoint)

    def run(self, fetches, feed_dict = None):
        '''
        Delegate to tf.Session.run
        :param fetches: operations to be run
        :param feed_dict: feed inputs into tensors in the graph
        :return: 
        '''
        if feed_dict is None:
            return self.sess.run(fetches)
        else:
            return self.sess.run(fetches, feed_dict=feed_dict)

    def __enter__(self):
        '''
        for the beggining of the 'with' statement
        :return: self 
        '''
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        '''
        for the end of the 'with' statement
        :param exc_type: 
        :param exc_val: 
        :param exc_tb: 
        :return: 
        '''
        # close self.sess
        if self.sess is not None:
            # auto-save current state before session is closed
            #  down
            if self._auto_save:
                self.save(self._check_point)
                print('--------------------------------------')
                print('Check point is saved to {}'.format(self._check_point))
            self.sess.close()
            self.sess = None

import os
import tensorflow as tf

class Session:
    '''
    A thin wrapper class for tf.Session allows loading/saving more easily
    '''
    def __init__(self, graph, max_to_keep=50):
        self.graph = graph

        with graph.as_default():
            self.sess  = tf.Session(graph=graph)
            self.saver = tf.train.Saver(max_to_keep=max_to_keep)

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
            self.sess.close()
            self.sess = None

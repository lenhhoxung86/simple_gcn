'''
This is the script for building the GCN model. 
The code is based on https://github.com/tkipf/gcn and and https://github.com/dennybritz/cnn-text-classification-tf.
'''


import tensorflow as tf
import numpy as np

flags = tf.app.flags
FLAGS = flags.FLAGS

############################################## Utilities function #########################################
def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)

def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res

def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


############################################## GCN class ##################################################

class GCN(object):
    """docstring for GCN"""
    def __init__(self, learning_rate, num_input, num_classes, hidden_dimensions=[64], sparse_input=True, act=tf.nn.relu):
        # super(GCN, self).__init__()

        # Placeholders for input, output and dropout
        self.adj_hat = tf.sparse_placeholder(tf.float32, name='adjacency_matrix')
        self.input_x = tf.sparse_placeholder(tf.float32, shape=[None, num_input], name="input_x")
        self.oh_labels = tf.placeholder(tf.float32, shape=[None, num_classes], name="oh_labels")
        self.labels_mask = tf.placeholder(tf.float32, name="labels_mask")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.num_features_nonzero = tf.placeholder(tf.int32, name="num_features_nonzero")
        self.loss = 0.0
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        with tf.variable_scope("graph_convo"):
            # scope for graph convolutional layer
            x = self.input_x
            layer_input_dimensionality = num_input
            layer_sparse_input = sparse_input
            for indx,h_dimensionality in enumerate(hidden_dimensions):
                # dropout
                if layer_sparse_input:
                    x = sparse_dropout(x, self.dropout_keep_prob, self.num_features_nonzero)
                else:
                    x = tf.nn.dropout(x, self.dropout_keep_prob)

                # define the weight here
                W = tf.get_variable("WGC_"+str(indx), shape=[layer_input_dimensionality,h_dimensionality],initializer=tf.contrib.layers.xavier_initializer())

                # compute the first matrix multiplication between x and the weight
                pre_h = dot(x, W, sparse=layer_sparse_input)

                # compute the second matrix multiplication with A hat
                h = dot(self.adj_hat, pre_h, sparse=True)

                # add non-linearity
                x = act(h)                    

                # update parameter
                layer_input_dimensionality = h_dimensionality
                layer_sparse_input = False

                # compute the l2 loss for l2 regularization later
                self.loss += tf.nn.l2_loss(W)
            self.h_activation = x 


        with tf.variable_scope("output"):
            # scope for output
            W = tf.get_variable("WFC", shape=[hidden_dimensions[-1],num_classes],initializer=tf.contrib.layers.xavier_initializer())
            h = dot(self.h_activation, W)
            self.output = dot(self.adj_hat, h, sparse=True)
            self.output_prob = tf.nn.softmax(self.output)

            # compute the l2 loss for l2 regularization later
            self.loss = FLAGS.weight_decay * self.loss


        with tf.variable_scope("loss"):
            # Sum over the l2_loss and the cross entropy loss
            self.loss += masked_softmax_cross_entropy(self.output, self.oh_labels, self.labels_mask)
            self.opt_op = self.optimizer.minimize(self.loss)


        with tf.variable_scope("accuracy"):
            self.accuracy = masked_accuracy(self.output, self.oh_labels, self.labels_mask)






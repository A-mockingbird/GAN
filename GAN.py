import tensorflow as tf
import numpy
class GAN(object):
    def init(self, sess, imdb):
        self.sess = sess
        self.imdb = imdb
        self.names_class = names_class
        self.batch_size =batch_size
        
    def createGAN(self, config):
        self.number_classes = number_classes
        self.hidden_layer1 = hidden_layer1
        self.hidden_layer2 = hidden_layer2
        self.input_images = tf.placeholder(tf.float32, [None, self.input_size])
        self.keep = tf.placeholder(tf.float32)
        g_images = _build_generate_network(self.input_images)
    
    def _build_generate_network(self, g_input):
        generate_weights = _initialize_generate_variables()
        layer0 = tf.nn.relu(tf.matmul(g_input, generate_weights['w0']) + generate_weights['b0'])
        drop_layer0 = tf.nn.dropout(layer0, self.keep)
        layer1 = tf.nn.relu(tf.matmul(layer0, generate_weights['w1']) + generate_weights['b1'])
        drop_layer1 = tf.dropout(layer1, self.keep)
        output_layer = tf.nn.relu(tf.matmul(layer1, generate_weights['w2']) + generate_weights['b2'])
        drop_output_layer = tf.dropout(output_layer, self.keep)
        return drop_output_layer

    def _initialize_generate_variables(self):
        generate_weights = dict{}
        w0 = tf.Variable(tf.truncated_normal([self.input_size, self.hidden_layer1], stddev=0.1))
        generate_weights['w0'] = w0
        b0 = tf.Variable(tf.zeros([self.hidden_layer1]))
        generate_weights['b0'] = b0
        w1 = tf.Variable(tf.zeros([self.hidden_layer1], self.hidden_layer2))
        generate_weights['w1'] = w1
        b1 = tf.Variable(tf.zeros([self.hidden_layer2]))
        generate_weights['b1'] = b1
        w2 = tf.Variable(tf.zeros([self.hidden_layer2, self.input_images]))
        generate_weights['w2'] = w2
        b2 = tf.Variable(tf.zeros([self.input_images]))
        generate_weights['b2'] = b2
        return generate_weights
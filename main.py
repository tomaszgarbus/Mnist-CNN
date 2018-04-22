import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from math import sqrt
from typing import Iterable, List
import numpy as np

# TODO:
# * achieve accuracy 99.1%
# * add batch normalization
# * find 10 patches that excite the network the most


class MnistTrainer:
    def __init__(self,
                 num_conv_layers: int = 1,
                 dense_layers: Iterable[int] = [64, 64, 10],
                 kernel_size: [int, int] = [5, 5],
                 filters: int = 10) -> None:
        self.num_conv_layers = num_conv_layers
        self.dense_layers = dense_layers
        self.kernel_size = kernel_size
        self.filters = filters
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    def create_model(self) -> None:
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')

        # Reshape input for convolution layers
        signal = tf.reshape(self.x, [-1, 28, 28, 1])

        # Apply convolution layers
        for layer_no in range(self.num_conv_layers):
            # Initialize weights with st. dev = sqrt(2/N)
            input_size = int(signal.get_shape()[1])
            w_init = tf.initializers.random_normal(stddev=sqrt(2 / input_size))

            # Build convolution layer
            cur_conv_layer = tf.layers.conv2d(inputs=signal,
                                              filters=self.filters,
                                              kernel_size=self.kernel_size,
                                              padding='same',
                                              activation=None,
                                              kernel_initializer=w_init,
                                              use_bias=False)  # with batch normalization, we don't need bias

            # Do batch normalization
            #   * Calculate mean.
            mb_mean = tf.reduce_mean(cur_conv_layer, axis=[0, 1, 2], keepdims=True)
            #   * Calculate variance and stdev.
            mb_variance = tf.reduce_mean(tf.multiply(cur_conv_layer-mb_mean, cur_conv_layer-mb_mean))
            eps = tf.constant(0.001, dtype=tf.float32)
            mb_stdev = tf.sqrt(mb_variance + eps)
            #   * Normalize.
            normalized_conv_layer = (cur_conv_layer - mb_mean)/mb_stdev
            #   * Scale and shift. Beta and Gamma are learned variables.
            beta = tf.Variable(initial_value=tf.ones([1, 1, 1, self.filters], dtype=tf.float32))
            gamma = tf.Variable(initial_value=tf.ones([1, 1, 1, self.filters], dtype=tf.float32))
            cur_conv_layer = normalized_conv_layer * beta + gamma

            # Apply activation to convolved (and batch normalized) signal.
            cur_conv_layer = tf.nn.relu(cur_conv_layer)

            # Apply max pooling, reducing image dimensions in half.
            cur_pool = tf.layers.max_pooling2d(inputs=cur_conv_layer, pool_size=[2, 2], strides=2)


            # Find exciting patches
            for filter_no in range(self.filters):
                inp = 28 // (2**layer_no)
                single_filtered_flattened = tf.reshape(cur_conv_layer[:, :, :, filter_no], [-1])
                top10_vals, top10_indices = tf.nn.top_k(single_filtered_flattened,
                                                        k=10,
                                                        sorted=True)
                top10_reshaped = tf.map_fn(lambda i: [i//(inp**2), (i//inp) % inp, i % inp],
                                           top10_indices,
                                           dtype=[tf.int32, tf.int32, tf.int32])
                def safe_cut_patch(a):
                    sample_no, x, y = a
                    padding = [[0, 0],
                               [0, self.kernel_size[0]],
                               [0, self.kernel_size[1]],
                               [0, 0]]
                    padded = tf.pad(signal, padding)
                    return padded[sample_no, x:x+self.kernel_size[0], y:y+self.kernel_size[1], :]
                exciting_patches = tf.map_fn(safe_cut_patch,
                                             top10_reshaped,
                                             dtype=tf.float32)
                if layer_no == 0:
                    tf.summary.image("exciting_patches_{0}_{1}".format(layer_no, filter_no),
                        tf.reshape(exciting_patches, [1] + [self.kernel_size[0], 10 * self.kernel_size[1]] + [1]))
                else:
                    tf.summary.image("exciting_patches_{0}_{1}".format(layer_no, filter_no),
                        tf.reshape(exciting_patches,
                                   [1] + [self.filters * self.kernel_size[0], 10 * self.kernel_size[1]] + [1]))
                self.exciting_patches = exciting_patches, top10_vals

            # Set pooled image as new signal
            signal = cur_pool

            # Write summaries
            for i in range(self.filters):
                if layer_no == 0:
                    tmp_str = 'conv2d/kernel:0'
                else:
                    tmp_str = 'conv2d_%d/kernel:0' % layer_no
                kernel = [v for v in tf.global_variables() if v.name == tmp_str][0]
                kernel = kernel[:, :, :, i]
                if layer_no == 0:
                    kernel = tf.reshape(kernel, [-1] + self.kernel_size + [1])
                    multed = tf.reshape(cur_conv_layer[0, :, :, i], [-1, inp, inp, 1])
                else:
                    kernel = tf.reshape(kernel, [-1] + [self.kernel_size[0], self.kernel_size[1] * self.filters] + [1])
                    multed = tf.reshape(cur_conv_layer[0, :, :, i], [-1, inp, inp, 1])
                tf.summary.image('conv%d%d_kernel' % (layer_no, i),
                                 kernel,
                                 max_outputs=3)
                tf.summary.image('conv%d%d_multiplied' % (layer_no, i),
                                 multed,
                                 max_outputs=3)

        # Flatten for the use in dense layers
        if self.num_conv_layers > 0:
            signal = tf.reshape(signal, [-1, self.filters * (28//(2**self.num_conv_layers))**2])
        else:
            signal = tf.reshape([-1, 784])

        # Apply dense layers with ReLU activation
        for num_neurons in self.dense_layers[:-1]:
            # Initialize weights and biases with stdev = sqrt(2/N)
            input_size = int(signal.get_shape()[1])
            w_init = tf.initializers.random_normal(stddev=sqrt(2/input_size))
            b_init = tf.initializers.constant(0)

            cur_layer = tf.layers.dense(inputs=signal,
                                        units=num_neurons,
                                        activation=tf.nn.relu,
                                        kernel_initializer=w_init,
                                        bias_initializer=b_init)
            signal = cur_layer

        # Apply last dense layer
        cur_layer = tf.layers.dense(inputs=signal,
                                    units=self.dense_layers[-1])
        signal = cur_layer

        self.merged_summary = tf.summary.merge_all()

        # Compute loss and accuracy.
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=signal, labels=self.y))
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, axis=1), tf.argmax(signal, axis=1)), tf.float32))

        # Use some optimizer from tf.
        self.train_op = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9).minimize(self.loss)

        print('list of variables', list(map(lambda x: x.name, tf.global_variables())))

    def train_on_batch(self, batch_x, batch_y, global_step) -> List:
        results = self.sess.run([self.loss, self.accuracy, self.train_op], #self.merged_summary,
                                feed_dict={self.x: batch_x, self.y: batch_y})
        # Update summary every 100 steps.
        if global_step % 100 == 0:
           writer = tf.summary.FileWriter(logdir='summary')
           writer.add_summary(results[2], global_step=global_step)
        return results[:2]

    def test_on_batch(self, batch_x, batch_y) -> List:
        # Note that this function does not fetch |self.train_op|, so that the weights
        # are not updated.
        results = self.sess.run([self.loss, self.accuracy, self.exciting_patches],
                                feed_dict={self.x: batch_x, self.y: batch_y})
        return results[:2]

    def test_on_all(self) -> List:
        """ Tests model on all images """
        x_test, y_test = self.mnist.test.images, self.mnist.test.labels
        N = self.mnist.test.num_examples
        results = np.array([0., 0.])
        for batch_no in range(N // self.mb_size):
            beg = batch_no * self.mb_size
            end = min(N, (batch_no + 1) * self.mb_size)
            len_batch = end - beg
            batch_results = np.array(self.test_on_batch(x_test[beg:end], y_test[beg:end]))
            results += batch_results * len_batch
        results /= N
        print("TEST: ", results)

    def train(self) -> None:
        with tf.Session() as self.sess:
            # Set mini-batch size and epochs number
            self.mb_size = 512
            nb_epochs = 50000
            self.learning_rate = 0.2

            # Initialize computation graph
            self.create_model()

            # Initialize variables
            tf.global_variables_initializer().run()

            for epoch_no in range(nb_epochs):
                batch_x, batch_y = self.mnist.train.next_batch(self.mb_size)
                results = self.train_on_batch(batch_x, batch_y, global_step=epoch_no)
                # Learning rate decay
                if epoch_no % 3000 == 0:
                    self.test_on_all()
                    self.learning_rate /= 2
                if epoch_no % 100 == 0:
                    print(epoch_no, ": ", results)
                    # Validate on test batch
                    batch_x_t, batch_y_t = self.mnist.test.next_batch(self.mb_size)
                    test_results = self.test_on_batch(batch_x_t, batch_y_t)
                    print("TEST: ", epoch_no, ": ", test_results)
            self.test_on_all()


if __name__ == '__main__':
    trainer = MnistTrainer(
        num_conv_layers = 2,
        dense_layers = [64, 64, 128] + [10],
        kernel_size=[5, 5],
        filters=10
    )
    trainer.train()

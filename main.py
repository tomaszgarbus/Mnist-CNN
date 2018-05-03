import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from math import sqrt
from typing import Iterable, List
import numpy as np
import logging

# TODO:
# * achieve accuracy 99.1%
# * add batch normalization (DONE)
# * find 10 patches that excite the network the most (DONE)


class MnistTrainer:
    def __init__(self,
                 conv_layers: Iterable[int] = [10, 10],
                 dense_layers: Iterable[int] = [64, 64, 10],
                 kernel_size: [int, int] = [5, 5],
                 nb_epochs: int = 100000,
                 learning_rate: int = 0.2,
                 lr_decay_time: int = 10000,
                 mb_size: int = 100,
                 dropout_rate: float = 0.4) -> None:
        # Number of filters in each convolutional layer.
        self.conv_layers = conv_layers
        # Number of neurons in each dense layer.
        self.dense_layers = dense_layers
        # Kernel size, common for convolutional layers.
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs
        # Each |lr_decay_time| epochs, learning rate is divided by 2.
        self.lr_decay_time = lr_decay_time
        # Mini batch size.
        self.mb_size = mb_size
        # Dropout after each dense layer (excluding last).
        self.dropout_rate = dropout_rate

        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

        # Initialize logging.
        self.logger = logging.Logger("main_logger", level=logging.INFO)
        log_file = 'log.txt'
        formatter = logging.Formatter(
            fmt='{levelname:<7} {message}',
            style='{'
        )
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def create_model(self) -> None:
        self.x = tf.placeholder(tf.float32, [self.mb_size, 784], name='x')
        self.y = tf.placeholder(tf.float32, [self.mb_size, 10], name='y')

        # Initialize fetch handles for exciting patches and their respective responses.
        self.exciting_patches = [[None] * k for k in self.conv_layers]
        self.patches_responses = [[None] * k for k in self.conv_layers]
        self.flattened_exciting_patches = [[None] * k for k in self.conv_layers]
        self.all_exciting_patches_at_layer = [None for _ in self.conv_layers]

        # Reshape input for convolution layers
        signal = tf.reshape(self.x, [self.mb_size, 28, 28, 1])

        # Apply convolution layers
        for layer_no in range(len(self.conv_layers)):
            # Initialize weights with st. dev = sqrt(2/N
            input_size = int(signal.get_shape()[1])
            w_init = tf.initializers.random_normal(stddev=sqrt(2 / input_size))

            # Build convolution layer
            cur_conv_layer = tf.layers.conv2d(inputs=signal,
                                              filters=self.conv_layers[layer_no],
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
            beta = tf.Variable(initial_value=tf.ones([1, 1, 1, self.conv_layers[layer_no]], dtype=tf.float32))
            gamma = tf.Variable(initial_value=tf.ones([1, 1, 1, self.conv_layers[layer_no]], dtype=tf.float32))
            cur_conv_layer = normalized_conv_layer * beta + gamma

            # Apply activation to convolved (and batch normalized) signal.
            cur_conv_layer = tf.nn.relu(cur_conv_layer)

            # Apply max pooling, reducing image dimensions in half.
            cur_pool = tf.layers.max_pooling2d(inputs=cur_conv_layer, pool_size=[2, 2], strides=2)

            # Find exciting patches
            for filter_no in range(self.conv_layers[layer_no]):
                inp = 28 // (2**layer_no)  # Square root of input shape for current layer.

                # Find top 10 responses to current filter, in the current mini-batch.
                single_filtered_flattened = tf.reshape(cur_conv_layer[:, :, :, filter_no],
                                                       [self.mb_size * inp * inp])
                top10_vals, top10_indices = tf.nn.top_k(single_filtered_flattened,
                                                        k=10,
                                                        sorted=True)
                top10_reshaped = tf.map_fn(lambda sxy: [sxy//(inp**2), (sxy//inp) % inp, sxy % inp],
                                           top10_indices,
                                           dtype=[tf.int32, tf.int32, tf.int32])

                # Find patches corresponding to the top 10 responses.
                def safe_cut_patch(a):
                    """
                    :param (sample_no, x, y)@a
                    :return: Cuts out a patch of size (convolution kernel) located at (x, y) on
                        input #sample_no in current batch.
                    """
                    sample_no, x, y = a
                    pad_marg_x = (self.kernel_size[0] // 2) + 1 - (self.kernel_size[0] % 2)
                    pad_marg_y = (self.kernel_size[1] // 2) + 1 - (self.kernel_size[1] % 2)
                    padding = [[0, 0],
                               [pad_marg_x, pad_marg_x],
                               [pad_marg_y, pad_marg_y],
                               [0, 0]]
                    padded = tf.pad(signal, padding)
                    return padded[sample_no, x:x+self.kernel_size[0], y:y+self.kernel_size[1], :]

                # Store patches and responses in class-visible array to be retrieved later.
                self.exciting_patches[layer_no][filter_no] = tf.map_fn(safe_cut_patch,
                                                                       top10_reshaped,
                                                                       dtype=tf.float32)
                self.patches_responses[layer_no][filter_no] = top10_vals

                # Flatten and concatenate the 10 patches to 2 dimensions for visualization.
                if layer_no == 0:
                    print(self.exciting_patches[layer_no][filter_no].get_shape())
                    flattened_patches_shape = [1] +\
                                              [10 * self.kernel_size[0],
                                               self.kernel_size[1]] +\
                                              [1]
                else:
                    print(self.exciting_patches[layer_no][filter_no].get_shape())
                    flattened_patches_shape = [1] +\
                                              [self.conv_layers[layer_no-1] * self.kernel_size[0],
                                               10 * self.kernel_size[1]] +\
                                              [1]
                # Write patches to summary.
                patch_name = "exciting_patches_filter{0}".format(filter_no)
                flattened_exciting_patches = tf.reshape(self.exciting_patches[layer_no][filter_no],
                                                        flattened_patches_shape,
                                                        name=patch_name)
                self.flattened_exciting_patches[layer_no][filter_no] = flattened_exciting_patches
                tf.summary.image(patch_name,
                                 flattened_exciting_patches,
                                 family='exciting_layer{0}'.format(layer_no))
            self.all_exciting_patches_at_layer[layer_no] = tf.concat(self.flattened_exciting_patches[layer_no], axis=2)
            # Write patches to summary.
            all_patches_name = "exciting_patches_layer{0}".format(layer_no)
            tf.summary.image(all_patches_name,
                             self.all_exciting_patches_at_layer[layer_no],
                             family='exciting_all_layers')

            # Set pooled image as new signal
            signal = cur_pool

            # Write 2 summaries for each filter:
            #  * kernel
            #  * input image with applied convolution
            for i in range(self.conv_layers[layer_no]):
                if layer_no == 0:
                    tmp_str = 'conv2d/kernel:0'
                else:
                    tmp_str = 'conv2d_%d/kernel:0' % layer_no
                kernel = [v for v in tf.global_variables() if v.name == tmp_str][0]
                print(kernel.get_shape())
                kernel = kernel[:, :, :, i]
                if layer_no == 0:
                    kernel = tf.reshape(kernel, [1] + self.kernel_size + [1])
                    applied = tf.reshape(cur_conv_layer[0, :, :, i], [1, inp, inp, 1])
                else:
                    kernel = tf.reshape(kernel, [1] +\
                                                [self.kernel_size[0],
                                                 self.kernel_size[1] * self.conv_layers[layer_no-1]] + [1])
                    applied = tf.reshape(cur_conv_layer[0, :, :, i], [1, inp, inp, 1])
                tf.summary.image('conv{0}_filter{1}_kernel'.format(layer_no, i),
                                 kernel,
                                 family='kernels_layer{0}'.format(layer_no),
                                 max_outputs=1)
                tf.summary.image('conv{0}_filter{1}_applied'.format(layer_no, i),
                                 applied,
                                 family='convolved_layer_{0}'.format(layer_no),
                                 max_outputs=1)

        # Flatten the signal for the use in dense layers
        if len(self.conv_layers) > 0:
            signal = tf.reshape(signal,
                                [self.mb_size, self.conv_layers[-1] * (28 // (2 ** len(self.conv_layers))) ** 2])
        else:
            signal = tf.reshape([self.mb_size, 784])

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

            signal = tf.layers.dropout(signal, rate=self.dropout_rate)

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
        self.train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)

        self.logger.info('list of variables {0}'.format(list(map(lambda x: x.name, tf.global_variables()))))

    def train_on_batch(self, batch_x, batch_y, global_step) -> List:
        """
        :return: [loss, accuracy]
        """
        # Update summary every 1000 steps.
        if global_step % 1000 == 0:
            results = self.sess.run([self.loss,
                                     self.accuracy,
                                     self.merged_summary,
                                     self.train_op],
                                    feed_dict={self.x: batch_x, self.y: batch_y})
            self.writer.add_summary(results[2], global_step=global_step)
        else:
            results = self.sess.run([self.loss, self.accuracy, self.train_op],
                                    feed_dict={self.x: batch_x, self.y: batch_y})
        return results[:2]

    def test_on_batch(self, batch_x, batch_y) -> List:
        """
        Note that this function does not fetch |self.train_op|, so that the weights
        are not updated.
        :param batch_x:
        :param batch_y:
        :return: [loss, accuracy]
        """
        results = self.sess.run([self.loss, self.accuracy, self.exciting_patches],
                                feed_dict={self.x: batch_x, self.y: batch_y})
        return results[:2]

    def test_on_all(self) -> None:
        """
        Tests model on all images
        :return: None, result is just printed.
        """
        x_test, y_test = self.mnist.test.images, self.mnist.test.labels
        N = self.mnist.test.num_examples

        # I have replaced all -1 with self.mb_size to be sure about sizes of all layers.
        assert N % self.mb_size == 0,\
            "Sorry, mb_size must divide the number of images in test set"

        results = np.array([0., 0.])
        for batch_no in range(N // self.mb_size):
            beg = batch_no * self.mb_size
            end = min(N, (batch_no + 1) * self.mb_size)
            len_batch = end - beg
            batch_results = np.array(self.test_on_batch(x_test[beg:end], y_test[beg:end]))
            results += batch_results * len_batch
        results /= N
        self.logger.info("(Test(final):   Loss: {0[0]}, accuracy: {0[1]}".format(results))

    def train_and_evaluate(self) -> None:
        """
        Train and evaluate model.
        """
        with tf.Session() as self.sess:
            # Initialize computation graph.
            self.create_model()

            # Initialize variables.
            tf.global_variables_initializer().run()

            # Initialize summary writer.
            self.writer = tf.summary.FileWriter(logdir='conv_vis')

            for epoch_no in range(self.nb_epochs):
                # Train model on next batch
                batch_x, batch_y = self.mnist.train.next_batch(self.mb_size)
                results = self.train_on_batch(batch_x, batch_y, global_step=epoch_no)

                if epoch_no > 0 and epoch_no % self.lr_decay_time == 0:
                    # Test on all samples.
                    self.test_on_all()
                    # Perform learning rate decay.
                    self.learning_rate /= 2
                if epoch_no % 100 == 0:
                    self.logger.info("Epoch {0}: Loss: {1[0]}, accuracy: {1[1]}".format(epoch_no, results))
                    batch_x_t, batch_y_t = self.mnist.test.next_batch(self.mb_size)
                    test_results = self.test_on_batch(batch_x_t, batch_y_t)
                    self.logger.info("(Test(batch):   Loss: {0[0]}, accuracy: {0[1]}".format(test_results))
            self.test_on_all()

            # Save the trained model with all valuable variables.
            saver = tf.train.Saver()
            saver.save(sess=self.sess, save_path='./saved_model', global_step=epoch_no)


if __name__ == '__main__':
    trainer = MnistTrainer(
        conv_layers=[10, 20],
        dense_layers=[64, 128] + [10],
        kernel_size=[5, 5],
        nb_epochs=100000
    )
    trainer.train_and_evaluate()

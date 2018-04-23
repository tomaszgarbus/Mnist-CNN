import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from math import sqrt
from typing import Iterable, List
import numpy as np

# TODO:
# * achieve accuracy 99.1%
# * add batch normalization (DONE)
# * find 10 patches that excite the network the most


class MnistTrainer:
    def __init__(self,
                 num_conv_layers: int = 1,
                 dense_layers: Iterable[int] = [64, 64, 10],
                 kernel_size: [int, int] = [5, 5],
                 num_filters: int = 10) -> None:
        self.num_conv_layers = num_conv_layers
        self.dense_layers = dense_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        self.mb_size = 200
        self.learning_rate = 0.2

    def create_model(self) -> None:
        self.x = tf.placeholder(tf.float32, [self.mb_size, 784], name='x')
        self.y = tf.placeholder(tf.float32, [self.mb_size, 10], name='y')

        # Initialize fetch handles for exciting patches and their respective responses.
        self.exciting_patches = [[None for _ in range(self.num_filters)] for _ in range(self.num_conv_layers)]
        self.patches_responses = [[None for _ in range(self.num_filters)] for _ in range(self.num_conv_layers)]

        # Reshape input for convolution layers
        signal = tf.reshape(self.x, [self.mb_size, 28, 28, 1])

        # Apply convolution layers
        for layer_no in range(self.num_conv_layers):
            # Initialize weights with st. dev = sqrt(2/N
            input_size = int(signal.get_shape()[1])
            w_init = tf.initializers.random_normal(stddev=sqrt(2 / input_size))

            # Build convolution layer
            cur_conv_layer = tf.layers.conv2d(inputs=signal,
                                              filters=self.num_filters,
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
            beta = tf.Variable(initial_value=tf.ones([1, 1, 1, self.num_filters], dtype=tf.float32))
            gamma = tf.Variable(initial_value=tf.ones([1, 1, 1, self.num_filters], dtype=tf.float32))
            cur_conv_layer = normalized_conv_layer * beta + gamma

            # Apply activation to convolved (and batch normalized) signal.
            cur_conv_layer = tf.nn.relu(cur_conv_layer)

            # Apply max pooling, reducing image dimensions in half.
            cur_pool = tf.layers.max_pooling2d(inputs=cur_conv_layer, pool_size=[2, 2], strides=2)

            # Find exciting patches
            for filter_no in range(self.num_filters):
                inp = 28 // (2**layer_no)
                single_filtered_flattened = tf.reshape(cur_conv_layer[:, :, :, filter_no], [self.mb_size * inp * inp])
                top10_vals, top10_indices = tf.nn.top_k(single_filtered_flattened,
                                                        k=10,
                                                        sorted=True)
                top10_reshaped = tf.map_fn(lambda i: [i//(inp**2), (i//inp) % inp, i % inp],
                                           top10_indices,
                                           dtype=[tf.int32, tf.int32, tf.int32])

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
                self.exciting_patches[layer_no][filter_no] = tf.map_fn(safe_cut_patch,
                                                                       top10_reshaped,
                                                                       dtype=tf.float32)
                self.patches_responses[layer_no][filter_no] = top10_vals
                if layer_no == 0:
                    new_shape = [1] + [self.kernel_size[0], 10 * self.kernel_size[1]] + [1]
                else:
                    new_shape = [1] + [self.num_filters * self.kernel_size[0], 10 * self.kernel_size[1]] + [1]
                patch_name = "exciting_patches_{0}_{1}".format(layer_no, filter_no)
                self.exciting_patches[layer_no][filter_no] = tf.reshape(self.exciting_patches[layer_no][filter_no],
                                                                        new_shape,
                                                                        name=patch_name)
                tf.summary.image(patch_name,
                                 self.exciting_patches[layer_no][filter_no])

            # Set pooled image as new signal
            signal = cur_pool

            # Write 2 summaries for each filter:
            #  * kernel
            #  * input image with applied convolution
            for i in range(self.num_filters):
                if layer_no == 0:
                    tmp_str = 'conv2d/kernel:0'
                else:
                    tmp_str = 'conv2d_%d/kernel:0' % layer_no
                kernel = [v for v in tf.global_variables() if v.name == tmp_str][0]
                kernel = kernel[:, :, :, i]
                if layer_no == 0:
                    kernel = tf.reshape(kernel, [1] + self.kernel_size + [1])
                    applied = tf.reshape(cur_conv_layer[0, :, :, i], [1, inp, inp, 1])
                else:
                    kernel = tf.reshape(kernel, [1] + [self.kernel_size[0], self.kernel_size[1] * self.num_filters] + [1])
                    applied = tf.reshape(cur_conv_layer[0, :, :, i], [1, inp, inp, 1])
                tf.summary.image('conv{0}_filter{1}_kernel'.format(layer_no, i),
                                 kernel,
                                 max_outputs=3)
                tf.summary.image('conv{0}_filter{1}_applied'.format(layer_no, i),
                                 applied,
                                 max_outputs=3)

        # Flatten for the use in dense layers
        if self.num_conv_layers > 0:
            signal = tf.reshape(signal, [self.mb_size, self.num_filters * (28 // (2 ** self.num_conv_layers)) ** 2])
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
        #self.train_op = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9).minimize(self.loss)

        print('list of variables', list(map(lambda x: x.name, tf.global_variables())))

    def train_on_batch(self, batch_x, batch_y, global_step) -> List:
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
        # Note that this function does not fetch |self.train_op|, so that the weights
        # are not updated.
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
        print("(Test(final):   Loss: {0[0]}, accuracy: {0[1]}".format(results))

    def find_exciting_patches(self) -> np.ndarray:
        """
        :return: np array of shape (self.num_conv_layers, self.filters, 10), containing,
            for each filter of each convolutional layer, 10 most exciting patches among
            all training images.
        """
        x_train, y_train = self.mnist.train.images, self.mnist.train.labels
        N = self.mnist.train.num_examples
        assert N % self.mb_size == 0, \
            "Sorry, mb_size must divide the number of images in train set"
        top10 = [[[] for _ in range(self.num_filters)] for _ in range(self.num_conv_layers)]
        for batch_no in range(N // self.mb_size):
            beg = batch_no * self.mb_size
            end = min(N, (batch_no + 1) * self.mb_size)
            batch_x, batch_y = x_train[beg:end], y_train[beg:end]
            results = self.sess.run([self.patches_responses, self.exciting_patches],
                                    feed_dict={self.x: batch_x, self.y: batch_y})
            for layer_no in range(self.num_conv_layers):
                for filter_no in range(self.num_filters):
                    results[0][layer_no][filter_no] = results[0][layer_no][filter_no].reshape(10).tolist()
                    tmp = list(zip(results[0][layer_no][filter_no], results[1][layer_no][filter_no]))
                    top10[layer_no][filter_no] += tmp
                    top10[layer_no][filter_no].sort(key=lambda a: a[0], reverse=True)
                    top10[layer_no][filter_no] = top10[layer_no][filter_no][:10]
        for layer_no in range(N // self.num_conv_layers):
            for filter_no in range(N // self.num_filters):
                top10[layer_no][filter_no] = top10[layer_no][filter_no][1]

        return np.array(top10)

    def train(self) -> None:
        """
        Train and evaluate model.
        """
        with tf.Session() as self.sess:
            # Set number of epochs.
            nb_epochs = 10000

            # Initialize computation graph.
            self.create_model()

            # Initialize variables.
            tf.global_variables_initializer().run()

            # Initialize summary writer.
            self.writer = tf.summary.FileWriter(logdir='summary')

            for epoch_no in range(nb_epochs):
                # Train model on next batch
                batch_x, batch_y = self.mnist.train.next_batch(self.mb_size)
                results = self.train_on_batch(batch_x, batch_y, global_step=epoch_no)

                if epoch_no % 10000 == 0:
                    self.test_on_all()
                    # Perform learning rate decay.
                    self.learning_rate /= 2
                if epoch_no % 100 == 0:
                    print("Epoch {0}: Loss: {1[0]}, accuracy: {1[1]}".format(epoch_no, results))
                    batch_x_t, batch_y_t = self.mnist.test.next_batch(self.mb_size)
                    test_results = self.test_on_batch(batch_x_t, batch_y_t)
                    print("(Test(batch):   Loss: {0[0]}, accuracy: {0[1]}".format(test_results))
            self.find_exciting_patches()
            self.test_on_all()

            saver = tf.train.Saver()
            saver.save(sess=self.sess, save_path='./saved_model', global_step=epoch_no)


if __name__ == '__main__':
    trainer = MnistTrainer(
        num_conv_layers=2,
        dense_layers=[128] + [10],
        kernel_size=[5, 5],
        num_filters=10
    )
    trainer.train()

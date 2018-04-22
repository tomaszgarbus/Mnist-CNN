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
                 kernel_size: [int, int] = [10, 10],
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
        # Log input images
        tf.summary.image(name='signal', tensor=signal)

        # Apply convolution layers
        for layer_no in range(self.num_conv_layers):
            # Initialize weights with stdev=sqrt(2/N)
            input_size = int(signal.get_shape()[1])
            w_init = tf.initializers.random_normal(stddev=sqrt(2 / input_size))
            b_init = tf.initializers.constant(0)

            # Build convolution layer
            cur_conv_layer = tf.layers.conv2d(inputs=signal,
                                              filters=self.filters,
                                              kernel_size=self.kernel_size,
                                              padding='same',
                                              activation=tf.nn.relu,
                                              kernel_initializer=w_init,
                                              bias_initializer=b_init)

            # Apply average pooling, reducing image dimensions in half
            cur_pool = tf.layers.average_pooling2d(inputs=cur_conv_layer, pool_size=[2, 2], strides=2)


            self.exciting_patches = []
            # Find exciting patches
            for filter_no in range(self.filters):
                inp = 28 // (2**layer_no)
                single_filtered_flattened = tf.reshape(cur_conv_layer[:, :, :, filter_no], [-1])
                top10_vals, top10_indices = tf.nn.top_k(single_filtered_flattened,
                                    k=10,
                                    sorted=True)
                top10_reshaped = tf.map_fn(lambda i: [i//(inp**2), (i//inp)%inp, i%inp],
                                           top10_indices,
                                           dtype=[tf.int32, tf.int32, tf.int32])
                def fun(a):
                    sample_no, y, x = a
                    padding = [[0, 0],
                               [0, self.kernel_size[0]],
                               [0, self.kernel_size[1]],
                               [0, 0]]
                    padded = tf.pad(signal, padding)
                    return padded[sample_no, x:x+self.kernel_size[0], y:y+self.kernel_size[1], :]
                exciting_patches = tf.map_fn(fun,
                                             top10_reshaped,
                                             dtype=tf.float32)
                self.exciting_patches.append(exciting_patches)
                if layer_no == 0:
                    tf.summary.image("exciting_patches%d".format(filter_no), tf.reshape(exciting_patches[0,:,:,:], [1]+self.kernel_size+[1]))
                self.dupa = exciting_patches
                #self.dupa = top10_reshaped
                """max_pos_each_sample = tf.argmax(single_filtered_flattened, axis=1)
                num_samples = self.mb_size
                max_each_sample_applied = tf.map_fn(lambda i: single_filtered_flattened[i][max_pos_each_sample[i]],
                                                    np.array(list(range(0, num_samples)), dtype='int32'),
                                                    dtype='float')
                max_each_sample_applied = tf.reshape(max_each_sample_applied, [self.mb_size])
                # b = tf.argmax(a, axis=1)
                # c = tf.map_fn(lambda i: a[i][b[i]], np.array([0, 1, 2, 3], dtype='int32'))
                # d = tf.argmax(c, axis=0)

                self.dupa = #max_pos_each_sample
                max_each_sample_applied"""

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
                kernel = tf.reshape(kernel, [-1] + self.kernel_size + [1])
                multed = tf.reshape(cur_conv_layer[:, :, :, i], [-1, 28, 28, 1])
                tf.summary.image('conv%d%d_kernel' % (layer_no, i),
                                 kernel,
                                 max_outputs=3)
                tf.summary.image('conv%d%d_multiplied' % (layer_no, i),
                                 multed,
                                 max_outputs=3)

        # Flatten for the use in dense layers
        if self.num_conv_layers > 0:
            signal = tf.reshape(signal, [-1, 10 * (28//(2**self.num_conv_layers))**2])
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

        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=signal, labels=self.y))
        self.accuracy = tf.reduce_mean(
            tf.cast(tf.equal(tf.argmax(self.y, axis=1), tf.argmax(signal, axis=1)), tf.float32))

        # Use some magical optimizer from tf
        self.train_op = tf.train.MomentumOptimizer(0.05, momentum=0.9).minimize(self.loss)

        print('list of variables', list(map(lambda x: x.name, tf.global_variables())))

    def train_on_batch(self, batch_x, batch_y) -> List:
        results = self.sess.run([self.loss, self.accuracy, self.merged_summary, self.train_op, self.dupa],
                                feed_dict={self.x: batch_x, self.y: batch_y})
        writer = tf.summary.FileWriter(logdir='summary')
        writer.add_summary(results[2])
        return results[:2]

    def test_on_batch(self, batch_x, batch_y) -> List:
        # Note that this function does not fetch |self.train_op|, so that the weights
        # are not updated.
        results = self.sess.run([self.loss, self.accuracy, self.merged_summary],
                                feed_dict={self.x: batch_x, self.y: batch_y})
        return results[:-1]

    def train(self) -> None:
        with tf.Session() as self.sess:
            mb_size = 256
            nb_epochs = 8000
            self.create_model()
            tf.global_variables_initializer().run()
            for i in range(nb_epochs):
                batch_x, batch_y = self.mnist.train.next_batch(mb_size)
                results = self.train_on_batch(batch_x, batch_y)
                if i % 100 == 0:
                    print(i, ": ", results)
                    batch_x_t, batch_y_t = self.mnist.test.next_batch(mb_size)
                    test_results = self.test_on_batch(batch_x_t, batch_y_t)
                    print("TEST: ", i, ": ", test_results)
            batch_x_t, batch_y_t = self.mnist.test.images, self.mnist.test.labels
            test_results = self.test_on_batch(batch_x_t, batch_y_t)

            print("TEST: ", i, ": ", test_results)


if __name__ == '__main__':
    trainer = MnistTrainer(
        num_conv_layers = 1,
        dense_layers = [64] * 0 + [10]
    )
    trainer.train()

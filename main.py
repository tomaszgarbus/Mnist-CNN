import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from math import sqrt
from typing import Iterable, List

class MnistTrainer(object):
    def __init__(self,
                 num_conv_layers: int = 1,
                 dense_layers: Iterable[int] = [64, 64, 10]) -> None:
        self.num_conv_layers = num_conv_layers
        self.dense_layers = dense_layers

    def create_model(self) -> None:
        self.x = tf.placeholder(tf.float32, [None, 784], name='x')
        self.y = tf.placeholder(tf.float32, [None, 10], name='y')

        # Reshape input for convolution layers
        signal = tf.reshape(self.x, [-1, 28, 28, 1])
        tmpsum = tf.summary.image(name='signal', tensor=signal)

        # Apply convolution layers
        self.conv_layers = [None for _ in range(self.num_conv_layers)]
        for l in range(self.num_conv_layers):
            # Initialize weights with stdev=sqrt(2/N)
            input_size = int(signal.get_shape()[1])
            w_init = tf.initializers.random_normal(stddev=sqrt(2 / input_size))
            b_init = tf.initializers.constant(0)

            # Build convolution layer
            cur_conv_layer = tf.layers.conv2d(inputs=signal,
                                              filters=10,
                                              kernel_size=[10, 10],
                                              padding='same',
                                              activation=tf.nn.relu,
                                              kernel_initializer=w_init,
                                              bias_initializer=b_init)

            for i in range(10):
                if l == 0:
                    tmp_str = 'conv2d/kernel:0'
                else:
                    tmp_str = 'conv2d_%d/kernel:0' % l
                kernel = [v for v in tf.global_variables() if v.name == tmp_str][0]
                if l == 0:
                    tmp_str = 'conv2d/bias:0'
                else:
                    tmp_str = 'conv2d_%d/bias:0' % l
                bias = [v for v in tf.global_variables() if v.name == tmp_str][0]
                tf.summary.image('conv%d%d_kernel' % (l, i),
                                 tf.reshape(tf.slice(kernel,
                                          begin=[0, 0, 0, i],
                                          size=[10, 10, 1, 1]),
                                            [1, 10, 10, 1]
                                            ),
                                 max_outputs=3)

            self.conv_layers[l] = cur_conv_layer

            # Apply average pooling, reducing image dimensions in half
            cur_pool = tf.layers.average_pooling2d(inputs=cur_conv_layer, pool_size=[2, 2], strides=2)
            signal = cur_pool

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
        results = self.sess.run([self.loss, self.accuracy, self.merged_summary, self.train_op],
                                feed_dict={self.x: batch_x, self.y: batch_y})
        writer = tf.summary.FileWriter(logdir='summary')
        writer.add_summary(results[2])
        return results[:-2]

    def test_on_batch(self, batch_x, batch_y) -> List:
        results = self.sess.run([self.loss, self.accuracy, self.merged_summary],
                                feed_dict={self.x: batch_x, self.y: batch_y})
        return results[:-1]

    def train(self) -> None:
        with tf.Session() as self.sess:
            mb_size = 256
            nb_epochs = 1500
            self.create_model()
            tf.global_variables_initializer().run()
            self.mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
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

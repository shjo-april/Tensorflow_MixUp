
import numpy as np
import tensorflow as tf

from Define import *

kernel_initializer = tf.random_normal_initializer(mean = 0.0, stddev = 0.01, seed = None)
bias_initializer = tf.constant_initializer(value = 0.0)

def bn_relu_conv(x, filters, kernel_size, strides, padding, is_training, scope, bn = True, activation = True, use_bias = False):
    with tf.variable_scope(scope):
        if bn: x = tf.layers.batch_normalization(x, training = is_training, name = 'bn')
        if activation: x = tf.nn.leaky_relu(x, alpha = 0.1, name = 'leaky_relu')
        x = tf.layers.conv2d(inputs = x, filters = filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_initializer = kernel_initializer, use_bias = use_bias, name = 'conv2d')
    return x

def get_getter(ema):
    def ema_getter(getter, name, *args, **kwargs):
        var = getter(name, *args, **kwargs)
        ema_var = ema.average(var)
        return ema_var if ema_var else var
    return ema_getter

def ResNetBlock(x, is_training, layers, strides, channel, scope):
    with tf.variable_scope(scope):
        x = bn_relu_conv(x, channel, [3, 3], strides, 'same', is_training, 'conv0')

        for i in range(layers):
            out = bn_relu_conv(x, channel, [3, 3], 1, 'same', is_training, 'conv{}/1'.format(i))
            out = bn_relu_conv(out, channel, [3, 3], 1, 'same', is_training, 'conv{}/2'.format(i))
            x = x + out
    return x

def Global_Average_Pooling(x):
    _, h, w, c = x.shape.as_list()
    x = tf.layers.average_pooling2d(inputs = x, pool_size = [w, h], strides = 1)
    x = tf.layers.flatten(x)
    return x

def WideResNet(input_var, is_training, depth = 28, widen_factor = 2, init_channel = 64, reuse = False, getter = None):
    n = (depth - 4) // 6 # layers : 28 = n : 4
    channels = [init_channel, 
                init_channel * widen_factor, 
                init_channel * 2 * widen_factor, 
                init_channel * 4 * widen_factor]
    
    with tf.variable_scope('Wider-ResNet-28', reuse = reuse, custom_getter = getter):
        # 1. input = 32x32
        x = input_var / 127.5 - 1
        x = bn_relu_conv(x, channels[0], [3, 3], 1, 'same', is_training, 'conv1')

        # block 1 : 32x32
        x = ResNetBlock(x, is_training, n, 1, channels[1], 'Block1')
        print(x)

        # block 2 : 16x16
        x = ResNetBlock(x, is_training, n, 2, channels[2], 'Block2')
        print(x)

        # block 3 : 8x8
        x = ResNetBlock(x, is_training, n, 2, channels[3], 'Block3')
        print(x)

        x = tf.layers.batch_normalization(x, training = is_training, name = 'bn')
        x = tf.nn.leaky_relu(x, alpha = 0.1, name = 'leaky_relu')
        x = Global_Average_Pooling(x)

        logits = tf.layers.dense(x, units = CLASSES)
        predictions = tf.nn.softmax(logits, axis = -1)

    return logits, predictions

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [None, 32, 32, 3])
    logits, predictions = WideResNet(input_var, False)

    print(logits, predictions)

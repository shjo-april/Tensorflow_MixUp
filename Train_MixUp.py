# Copyright (C) 2019 * Ltd. All rights reserved.
# author : SangHyeon Jo <josanghyeokn@gmail.com>

import cv2
import time
import numpy as np
import tensorflow as tf

from Define import *
from WideResNet import *
from Utils import *
from DataAugmentation import *

# 1. dataset
train_data_list = np.load('./dataset/train_all.npy', allow_pickle = True)
test_data_list = np.load('./dataset/test.npy', allow_pickle = True)

test_iteration = len(test_data_list) // BATCH_SIZE

# 2. model
input_var = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL])
label_var = tf.placeholder(tf.float32, [None, CLASSES])
is_training = tf.placeholder(tf.bool)

logits_op, predictions_op = WideResNet(input_var, is_training)

loss_op = tf.nn.softmax_cross_entropy_with_logits(logits = logits_op, labels = label_var)
loss_op = tf.reduce_mean(loss_op)

vars = tf.trainable_variables()
l2_reg_loss_op = tf.add_n([tf.nn.l2_loss(var) for var in vars]) * WEIGHT_DECAY
loss_op += l2_reg_loss_op

correct_op = tf.equal(tf.argmax(predictions_op, axis = -1), tf.argmax(label_var, axis = -1))
accuracy_op = tf.reduce_mean(tf.cast(correct_op, tf.float32)) * 100

learning_rate_var = tf.placeholder(tf.float32)
with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    train_op = tf.train.AdamOptimizer(learning_rate_var).minimize(loss_op)

train_summary_dic = {
    'Loss/Total_Loss' : loss_op,
    'Loss/L2_Regularization_Loss' : l2_reg_loss_op,
    'Accuracy/Train' : accuracy_op,
    'Learning_rate' : learning_rate_var,
}

train_summary_list = []
for name in train_summary_dic.keys():
    value = train_summary_dic[name]
    train_summary_list.append(tf.summary.scalar(name, value))
train_summary_op = tf.summary.merge(train_summary_list)

test_accuracy_var = tf.placeholder(tf.float32)
test_accuracy_op = tf.summary.scalar('Accuracy/Test', test_accuracy_var)

# 3. train
sess = tf.Session()
sess.run(tf.global_variables_initializer())

learning_rate = INIT_LEARNING_RATE
log_print('[i] max_iteration : {}'.format(MAX_ITERATION))

train_writer = tf.summary.FileWriter('./logs/train_100%_with_MixUp')
train_ops = [train_op, loss_op, l2_reg_loss_op, accuracy_op, train_summary_op]
train_time = time.time()

# MixUp Function
def MixUp(images, labels):
    indexs = np.random.permutation(BATCH_SIZE)
    alpha = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA, BATCH_SIZE)

    image_alpha = alpha.reshape((BATCH_SIZE, 1, 1, 1))
    label_alpha = alpha.reshape((BATCH_SIZE, 1))

    x1, x2 = images, images[indexs]
    y1, y2 = labels, labels[indexs]

    images = image_alpha * x1 + (1 - image_alpha) * x2
    labels = label_alpha * y1 + (1 - label_alpha) * y2

    return images, labels

for iter in range(1, MAX_ITERATION + 1):
    if iter in DECAY_ITERATIONS:
        learning_rate /= 10.

    np.random.shuffle(train_data_list)
    batch_data_list = train_data_list[:BATCH_SIZE]
    
    batch_image_data = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), dtype = np.float32)
    batch_label_data = np.zeros((BATCH_SIZE, CLASSES), dtype = np.float32)
    
    for i, (_image, label) in enumerate(batch_data_list):
        image = _image.copy()
        image = RandomPadandCrop(image)
        image = RandomFlip(image)

        batch_image_data[i] = image.astype(np.float32)
        batch_label_data[i] = label.astype(np.float32)
    
    batch_image_data, batch_label_data = MixUp(batch_image_data, batch_label_data)

    _feed_dict = {
        input_var : batch_image_data, 
        label_var : batch_label_data, 
        is_training : True,
        learning_rate_var : learning_rate
    }

    _, loss, l2_reg_loss, accuracy, summary = sess.run(train_ops, feed_dict = _feed_dict)
    train_writer.add_summary(summary, iter)

    if iter % 100 == 0:
        train_time = int(time.time() - train_time)

        print('[i] iter = {}, loss = {:.4f}, l2_reg_loss = {:.4f}, accuracy = {:.2f}, train_time : {}sec'.format(iter, loss, l2_reg_loss, accuracy, train_time))

        train_time = time.time()

    if iter % 2000 == 0:
        test_time = time.time()
        test_accuracy_list = []

        for i in range(test_iteration):
            batch_data_list = test_data_list[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]

            batch_image_data = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, IMAGE_CHANNEL), dtype = np.float32)
            batch_label_data = np.zeros((BATCH_SIZE, CLASSES), dtype = np.float32)
            
            for i, (image, label) in enumerate(batch_data_list):
                batch_image_data[i] = image.astype(np.float32)
                batch_label_data[i] = label.astype(np.float32)
            
            _feed_dict = {
                input_var : batch_image_data,
                label_var : batch_label_data,
                is_training : False
            }
            accuracy = sess.run(accuracy_op, feed_dict = _feed_dict)
            test_accuracy_list.append(accuracy)

        test_time = int(time.time() - test_time)
        test_accuracy = np.mean(test_accuracy_list)

        summary = sess.run(test_accuracy_op, feed_dict = {test_accuracy_var : test_accuracy})
        train_writer.add_summary(summary, iter)

        print('[i] iter = {}, test_accuracy = {:.2f}, test_time = {}sec'.format(iter, test_accuracy, test_time))

saver = tf.train.Saver()
saver.save(sess, './model/MixUp_100%.ckpt')

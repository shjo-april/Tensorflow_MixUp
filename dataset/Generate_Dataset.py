import cv2
import sys
import pickle
import numpy as np

# refer : https://www.cs.toronto.edu/~kriz/cifar.html
# data -- a 10000x3072 numpy array of uint8s. Each row of the array stores a 32x32 colour image. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image.
# labels -- a list of 10000 numbers in the range 0-9. The number at index i indicates the label of the ith image in the array data.
def get_data(file):
    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def one_hot(label, classes):
    v = np.zeros((classes), dtype = np.float32)
    v[label] = 1.
    return v

def get_data_list(file_paths):
    data_list = []
    for file_path in file_paths:
        data = get_data(file_path)
        data_length = len(data[b'filenames'])

        for i in range(data_length):
            label = int(data[b'labels'][i])
            image_data = data[b'data'][i]

            channel_size = 32 * 32        

            r = image_data[:channel_size]
            g = image_data[channel_size : channel_size * 2]
            b = image_data[channel_size * 2 : ]

            r = r.reshape((32, 32)).astype(np.uint8)
            g = g.reshape((32, 32)).astype(np.uint8)
            b = b.reshape((32, 32)).astype(np.uint8)

            image = cv2.merge((b, g, r))
            label = one_hot(label, 10)

            data_list.append([image, label])
    return data_list

train_files = ['./dataset/cifar10/data_batch_{}'.format(i) for i in range(1, 5 + 1)]
test_files = ['./dataset/cifar10/test_batch']

train_data_list = get_data_list(train_files)
test_data_list = get_data_list(test_files)

np.save('./dataset/train_all.npy', train_data_list)
np.save('./dataset/test.npy', test_data_list)

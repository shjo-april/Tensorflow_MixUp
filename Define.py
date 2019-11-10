
IMAGE_SIZE = 32
IMAGE_CHANNEL = 3

'''
0 : airplane
1 : automobile
2 : bird
3 : cat
4 : deer
5 : dog
6 : frog
7 : horse
8 : ship
9 : truck
'''
CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
CLASSES = len(CLASS_NAMES)

T = 0.5
EMA_DECAY = 0.999
WEIGHT_DECAY = 0.0001

MIXUP_ALPHA = 0.2

INIT_LEARNING_RATE = 1e-4

MAX_ITERATION = 100000 # 100K
DECAY_ITERATIONS = [60000, 80000]
BATCH_SIZE = 64

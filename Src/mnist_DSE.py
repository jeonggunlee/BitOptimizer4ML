# MNIST and Deep learning CNN
import tensorflow as tf
import random
import numpy as np
import ctypes
# import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(777)  # reproducibility

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

def binRep(num, bits):
    binNum = bin(ctypes.c_uint.from_buffer(ctypes.c_float(num)).value)[2:]
    temp1 = binNum.rjust(32,"0")
    temp2 = temp1[0:bits]
    binNum = temp2.ljust(32,"0")
    #print("bits: " + binNum.rjust(32,"0"))
    mantissa = "1" + binNum[-23:]
    #print("sig (bin): " + mantissa.rjust(24))
    mantInt = int(mantissa,2)/2**23
    #print("sig (float): " + str(mantInt))
    base = int(binNum[-31:-23],2)-127
    #print("base:" + str(base))
    sign = 1-2*("1"==binNum[-32:-31].rjust(1,"0"))
    #print("sign:" + str(sign))
    #print("recreate:" + str(sign*mantInt*(2**base)))
    return sign*mantInt*(2**base)

# hyper parameters
learning_rate = 0.001
training_epochs = 15
batch_size = 100

# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing
keep_prob = tf.placeholder(tf.float32)

# input place holders
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, 10])

# L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
#    Conv     -> (?, 28, 28, 32)
#    Pool     -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
#L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],
                    strides=[1, 2, 2, 1], padding='SAME')
#L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

# L3 ImgIn shape=(?, 7, 7, 64)
W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
#    Conv      ->(?, 7, 7, 128)
#    Pool      ->(?, 4, 4, 128)
#    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[
                    1, 2, 2, 1], padding='SAME')
#L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])

# L4 FC 4x4x128 inputs -> 625 outputs
W4 = tf.get_variable("W4", shape=[128 * 4 * 4, 625],
                     initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)
#L4 = tf.nn.dropout(L4, keep_prob=keep_prob)

# L5 Final FC 625 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 10],
                     initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4, W5) + b5

# define cost/loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
base_path = './tf-models'
date_str = '201809'
load_path = base_path + '/' + date_str
save_path = load_path + '/my_model'
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

# restore model
saver.restore(sess, save_path)

# Test model and check accuracy

# if you have a OOM error, please refer to lab-11-X-mnist_deep_cnn_low_memory.py

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print('Accuracy:', sess.run(accuracy, feed_dict={
#      X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))

# Variables : Weights and Biases
w1_ = sess.run(W1)	# tf.random_normal([3, 3, 1, 32]    ~ 288
w2_ = sess.run(W2)	# tf.random_normal([3, 3, 32, 64]   ~ 18432
w3_ = sess.run(W3)	# tf.random_normal([3, 3, 64, 128]  ~ 73728
w4_ = sess.run(W4)	# shape=[128 * 4 * 4, 625]          ~ 1280000
b4_ = sess.run(b4)	# tf.random_normal([625])           ~ 625
w5_ = sess.run(W5)	# shape=[625, 10]                   ~ 6250
b5_ = sess.run(b5)	# tf.random_normal([10])            ~ 10

#threshold_list = [0, 0.001, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]
#bit_width_list = [10, 11, 12, 13, 14, 15, 16]
threshold_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
bit_width_list = [15, 14, 13, 12, 11, 10, 9]

ww1_ = np.zeros((3, 3, 1, 32))
ww2_ = np.zeros((3, 3, 32, 64))
ww3_ = np.zeros((3, 3, 64, 128))
ww4_ = np.zeros((128*4*4, 625))
ww5_ = np.zeros((625, 10))
bb4_ = np.zeros((625))
bb5_ = np.zeros((10))


for threshold in threshold_list:
    for BIT_WIDTH in bit_width_list:

        for i in range(3):
            for j in range(3):
                for k in range(1):
                    for l in range(32):
                        ww1_[i][j][k][l]=binRep(w1_[i][j][k][l], BIT_WIDTH)
        
        ww1_[abs(ww1_) < threshold] = 0
        _w1_ = ww1_.flatten()
        np.savetxt("w1_.txt", _w1_)

        for i in range(3):
            for j in range(3):
                for k in range(32):
                    for l in range(64):
                        ww2_[i][j][k][l]=binRep(w2_[i][j][k][l], BIT_WIDTH)

        ww2_[abs(ww2_) < threshold] = 0
        _w2_ = ww2_.flatten()
        np.savetxt("w2_.txt", _w2_)

        for i in range(3):
            for j in range(3):
                for k in range(64):
                    for l in range(128):
                        ww3_[i][j][k][l]=binRep(w3_[i][j][k][l], BIT_WIDTH)

        ww3_[abs(ww3_) < threshold] = 0
        _w3_ = ww3_.flatten()
        np.savetxt("w3_.txt", _w3_)

        for i in range(2048):
            for j in range(625):
                ww4_[i][j]=binRep(w4_[i][j], BIT_WIDTH)

        ww4_[abs(ww4_) < threshold] = 0
        _w4_ = ww4_.flatten()
        np.savetxt("w4_.txt", _w4_)

        for i in range(625):
            for j in range(10):
                ww5_[i][j]=binRep(w5_[i][j], BIT_WIDTH)

        ww5_[abs(ww5_) < threshold] = 0
        _w5_ = ww5_.flatten()
        np.savetxt("w5_.txt", _w5_)


        for i in range(625):
            bb4_[i]=binRep(b4_[i], BIT_WIDTH)

        bb4_[abs(bb4_) < threshold] = 0
        _b4_ = bb4_.flatten()
        np.savetxt("b4_.txt", _b4_)

        for i in range(10):
            bb5_[i]=binRep(b5_[i], BIT_WIDTH)

        bb5_[abs(bb5_) < threshold] = 0
        _b5_ = bb5_.flatten()
        np.savetxt("b5_.txt", _b5_)

        # All the data 
        all_p = np.concatenate((_w1_,  _w2_), axis=None)
        all_p = np.concatenate((all_p,  _w3_), axis=None)
        all_p = np.concatenate((all_p,  _w4_), axis=None)
        all_p = np.concatenate((all_p,  _w5_), axis=None)
        all_p = np.concatenate((all_p,  _b4_), axis=None)
        all_p = np.concatenate((all_p,  _b5_), axis=None)

        np.savetxt("all_p_opt_zero_elim.txt", all_p)

        zero_num = (all_p == 0).sum()

        # Reassign w1_ to W1
        sess.run(W1.assign(ww1_))
        sess.run(W2.assign(ww2_))
        sess.run(W3.assign(ww3_))
        sess.run(W4.assign(ww4_))
        sess.run(W5.assign(ww5_))
        sess.run(b4.assign(bb4_))
        sess.run(b5.assign(bb5_))
        # print(sess.run(W1))
				
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print(threshold, BIT_WIDTH, zero_num, sess.run(accuracy, feed_dict={
                 		X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
				


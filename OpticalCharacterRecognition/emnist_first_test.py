#from tensorflow.examples.tutorials.mnist import input_data # replace this with a way of importing emnist
emnist = input_data.read_data_sets('EMNIST_data', one_hot=True)

import matplotlib.pyplot as plt
import numpy as np
import random as ran
import tensorflow as tf

def TRAIN_SIZE(num):
    print ('Total Training Images in Dataset = ' + str(emnist.train.images.shape))
    print ('--------------------------------------------------')
    x_train = emnist.train.images[:num,:]
    print ('x_train Examples Loaded = ' + str(x_train.shape))
    y_train = emnist.train.labels[:num,:]
    print ('y_train Examples Loaded = ' + str(y_train.shape))
    print('')
    return x_train, y_train
    
def TEST_SIZE(num):
    print ('Total Test Examples in Dataset = ' + str(emnist.test.images.shape))
    print ('--------------------------------------------------')
    x_test = emnist.test.images[:num,:]
    print ('x_test Examples Loaded = ' + str(x_test.shape))
    y_test = emnist.test.labels[:num,:]
    print ('y_test Examples Loaded = ' + str(y_test.shape))
    return x_test, y_test
    
def display_digit(num):
    print(y_train[num])
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28,28])
    plt.title('Example: %d  Label: %d' % (num, label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()

def display_mult_flat(start, stop):
    images = x_train[start].reshape([1,784])
    for i in range(start+1,stop):
        images = np.concatenate((images, x_train[i].reshape([1,784])))
    plt.imshow(images, cmap=plt.get_cmap('gray_r'))
    plt.show()

x_train, y_train = TRAIN_SIZE(55000)


display_digit(ran.randint(0, x_train.shape[0]))
display_mult_flat(0,400)

sess = tf.Session()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)
print(y)

sess.run(tf.global_variables_initializer())
#If using TensorFlow prior to 0.12 use:
#sess.run(tf.initialize_all_variables())

print(sess.run(y, feed_dict={x: x_train}))

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
j = [0.03, 0.03, 0.01, 0.9, 0.01, 0.01, 0.0025,0.0025, 0.0025, 0.0025]

k = [0,0,0,1,0,0,0,0,0,0]
np.multiply(np.log(j),k)

k = [0,0,1,0,0,0,0,0,0,0]
np.multiply(np.log(j),k)

x_test, y_test = TEST_SIZE(10000)

LEARNING_RATE = 0.1
TRAIN_STEPS = 2500
init = tf.global_variables_initializer()
sess.run(init)
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
for i in range(TRAIN_STEPS+1):
    sess.run(training, feed_dict={x: x_train, y_: y_train})
    if i%100 == 0:
        print('Training Step:' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={x: x_test, y_: y_test})) + '  Loss = ' + str(sess.run(cross_entropy, {x: x_train, y_: y_train})))

for i in range(10):
    plt.subplot(2, 5, i+1)
    weight = sess.run(W)[:,i]
    plt.title(i)
    plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)

plt.show

x_train, y_train = TRAIN_SIZE(1)

display_digit(0)

answer = sess.run(y, feed_dict={x: x_train})
print(answer)

answer.argmax()

def display_compare(num):
    # THIS WILL LOAD ONE TRAINING EXAMPLE
    x_train = emnist.train.images[num,:].reshape(1,784)
    y_train = emnist.train.labels[num,:]
    # THIS GETS OUR LABEL AS A INTEGER
    label = y_train.argmax()
    # THIS GETS OUR PREDICTION AS A INTEGER
    prediction = sess.run(y, feed_dict={x: x_train}).argmax()
    plt.title('Prediction: %d Label: %d' % (prediction, label))
    plt.imshow(x_train.reshape([28,28]), cmap=plt.get_cmap('gray_r'))
    plt.show()

display_compare(ran.randint(0, 55000))
display_compare(ran.randint(0, 55000))
display_compare(ran.randint(0, 55000))
display_compare(2)
display_compare(2)
display_compare(ran.randint(0, 55000))
display_compare(ran.randint(0, 55000))
display_compare(ran.randint(0, 55000))
display_compare(ran.randint(0, 55000))
display_compare(ran.randint(0, 55000))
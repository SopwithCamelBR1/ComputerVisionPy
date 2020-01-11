#Import MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt
import numpy as np
import random as ran
import tensorflow as tf

#Function that creates a training set of data
def TRAIN_SIZE(num):
    x_train = mnist.train.images[:num,:].reshape(-1, 28, 28, 1)
    y_train = mnist.train.labels[:num,:]
    return x_train, y_train

#Function that creates a testing set of data
def TEST_SIZE(num):
    x_test = mnist.test.images[:num,:].reshape(-1, 28, 28, 1)
    y_test = mnist.test.labels[:num,:]
    return x_test, y_test

#Create Training and Testing dataset
x_train, y_train = TRAIN_SIZE(5500)
x_test, y_test = TEST_SIZE(1000)   

#Placeholders


#Define Model Layers
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x) 

def maxpool2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],padding='SAME')

inputs = tf.placeholder("float", [None, 28,28,1])
target = tf.placeholder("float", [None, 10])
weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W3', shape=(4*4*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(128,10), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(10), initializer=tf.contrib.layers.xavier_initializer()),
}

def conv_net(x, weights, biases):  

    # here we call the conv2d function we had defined above and pass the input image x, weights wc1 and bias bc1.
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 14*14 matrix.
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    # here we call the conv2d function we had defined above and pass the input image x, weights wc2 and bias bc2.
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 7*7 matrix.
    conv2 = maxpool2d(conv2, k=2)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc3'])
    # Max Pooling (down-sampling), this chooses the max value from a 2*2 matrix window and outputs a 4*4.
    conv3 = maxpool2d(conv3, k=2)


    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv3, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Output, class prediction
    # finally we multiply the fully connected layer with the weights and add a bias term. 
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out
    
#Model hyperparameters
model_output = conv_net(inputs, weights, biases)

#Cost Function
#cost= tf.losses.mean_squared_error(target, model_output)
cost = tf.reduce_mean(-tf.reduce_sum(target * tf.log(model_output), reduction_indices=[1]))

#Learning Algorithms
LEARNING_RATE = 0.1
TRAIN_STEPS = 2500
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

#functions for calculating accuracy
correct_prediction = tf.equal(tf.argmax(model_output,1), tf.argmax(target,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Create TF session and initialise variables
sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Training the model
for i in range(TRAIN_STEPS+1):
    sess.run(training, feed_dict={inputs: x_train, target: y_train})
    if i%100 == 0:
        print('Training Step:' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy, feed_dict={inputs: x_test, target: y_test})) + '  Cost = ' + str(sess.run(cost, {inputs: x_train, target: y_train})))

#Testing the Model
def run_trained_model(num):
    test_num = ran.randint(0, x_test.shape[0])
    display_test_digit(test_num)    
    m_output = np.array(sess.run(model_output, {inputs: x_test[test_num].reshape(1, 784)}).reshape(10)).tolist()
    maxpos = m_output.index(max(m_output))
    print('Model Output: ' + str(m_output))
    print('Prediction: ' + str(maxpos))
    
for i in range(0,5):
    alphabet = ['A','B','C','D','E','F']
    print('\nTest ' + alphabet[i])
    run_trained_model(i)
    input("Press [enter] to continue.")
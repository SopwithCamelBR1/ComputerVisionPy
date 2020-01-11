#Import MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt
import numpy as np
import random as ran
import tensorflow as tf

#Function that creates a training set of data
def TRAIN_SIZE(num):
    x_train = mnist.train.images[:num,:]
    y_train = mnist.train.labels[:num,:]
    return x_train, y_train

#Function that creates a testing set of data
def TEST_SIZE(num):
    x_test = mnist.test.images[:num,:]
    y_test = mnist.test.labels[:num,:]
    return x_test, y_test

#Create Training and Testing dataset
x_train, y_train = TRAIN_SIZE(5500)
x_test, y_test = TEST_SIZE(1000)   

#Model hyperparameters
inputs = tf.placeholder(tf.float32, shape=[None, 784])
target = tf.placeholder(tf.float32, shape=[None, 10])
weights = tf.Variable(tf.zeros([784,10]))

#The Model:
model_output = tf.nn.sigmoid(tf.matmul(inputs,weights))

#Cost Function
cost= tf.losses.mean_squared_error(target, model_output)

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
def run_trained_model():
    test_num = ran.randint(0, x_test.shape[0])
    label_array=str(y_test[test_num])
    label_list  = np.array(y_test[test_num]).tolist()
    print('Label Array: ' + label_array)
    print('Label: ' + str(label_list.index(max(label_list))))   
    m_output = np.array(sess.run(model_output, {inputs: x_test[test_num].reshape(1, 784)}).reshape(10)).tolist()
    maxpos = m_output.index(max(m_output))
    print('Model Output: ' + str(m_output))
    print('Prediction: ' + str(maxpos))
    
for i in range(0,5):
    alphabet = ['A','B','C','D','E','F']
    print('\nTest ' + alphabet[i])
    run_trained_model()
    input("Press [enter] to continue.")
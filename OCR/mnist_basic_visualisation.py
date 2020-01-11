'''
    Opening the Black Box
    
    So we've created, trained, and tested the model. ANN's are often described as a 'black-box' - we can see what goes in and what comes out, but cannot see the inner workings of the model.
    For the larger ANNs used in modern applications this true for now. However there are increasingly programs and methods being developed to help us see inside.
    
    Our comparitively simple model however we certainly can take a stab at understanding it.
    
'''    

#Import MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import matplotlib.pyplot as plt
import numpy as np
import random as ran
#Import tensorflow - put this at the start of the code.
import tensorflow as tf
sess = tf.Session()


#Visualise Network
def show_network(train_num, title_s):
    plt.suptitle(title_s + str(train_num+1))
    grid = plt.GridSpec(6, 10, wspace=0.4, hspace=1.0)
    #Input
    plt.subplot(grid[0, 4])
    plt.title('Input')
    plt.imshow(mnist.train.images[train_num].reshape([28,28]), cmap=plt.get_cmap('gray_r'))
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    #Weights
    for i in range(10):
        plt.subplot(grid[1, i])
        weight = sess.run(W)[:,i]
        plt.title('W' + str(i))
        plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    #Softmax Input
    for i in range(10):
        plt.subplot(grid[2, i])
        weight = sess.run(W)[:,i]
        plt.title('S' + str(i) + ' Input')
        input_by_weight=np.matmul(weight.reshape([28,28]), mnist.train.images[train_num].reshape([28,28])) 
        plt.imshow(input_by_weight, cmap=plt.get_cmap('seismic'))
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    #Target
    for i in range(10):
        plt.subplot(grid[4, i])
        weight = sess.run(W)[:,i]
        plt.title('Target')
        target=mnist.train.labels[train_num].tolist()
        maxpos_t = target.index(max(target))
        if maxpos_t == i:
            plt.text(0.5, 0.5, target[i], fontsize=18, ha='center', color='green')
        else:
            plt.text(0.5, 0.5, target[i], fontsize=18, ha='center')
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    #Softmax output
    for i in range(10):
        plt.subplot(grid[3, i])
        weight = sess.run(W)[:,i]
        plt.title('S' + str(i) + ' Output')
        softmax_out=np.array(sess.run(y, feed_dict={x: mnist.train.images[i]}).reshape(10)).tolist()
        maxpos_o = softmax_out.index(max(softmax_out))
        if maxpos_o == i:
            if maxpos_o == maxpos_t:
                plt.text(0.5, 0.5, round(softmax_out[i], 4), fontsize=18, ha='center', color='green')
            else:
                plt.text(0.5, 0.5, round(softmax_out[i], 4), fontsize=18, ha='center', color='red')
        else:
            plt.text(0.5, 0.5, round(softmax_out[i], 4), fontsize=18, ha='center')
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    
    #Cost
    plt.subplot(grid[5, 4])
    plt.title('Cost')
    error = sess.run(cost, feed_dict={x: mnist.train.images[train_num], y_: mnist.train.labels[train_num]}) 
    plt.text(0.5, 0.5, round(error, 4), fontsize=18, ha='center')
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    
    plt.draw()
    plt.pause(1)
    #plt.show()
    
#Visualise weights
def show_weight_plot(train_num):
    plt.suptitle('Number of examples trained: ' + str(train_num+1))
    for i in range(10):
        plt.subplot(2, 5, i+1)
        weight = sess.run(W)[:,i]
        plt.title(i)
        plt.imshow(weight.reshape([28,28]), cmap=plt.get_cmap('seismic'))
        frame1 = plt.gca()
        frame1.axes.get_xaxis().set_visible(False)
        frame1.axes.get_yaxis().set_visible(False)
    plt.draw()
    plt.pause(0.01)

#run network - deep dive   
def run_network_dd(i):
    show_network(i, 'Before Training, Training Example: ' )
    #input("Press [enter] to continue.")
    sess.run(training, feed_dict={x: mnist.train.images[i], y_: mnist.train.labels[i]})
    show_network(i, 'After Training, Training Example: ' )
    #input("Press [enter] to continue.")
    
#run network - weight overview     
def run_weight_ov(i):
    sess.run(training, feed_dict={x: mnist.train.images[i], y_: mnist.train.labels[i]})
    show_weight_plot(i)
       
#Model variables
x = tf.placeholder(tf.float32, shape=[784])
x_reshape = tf.expand_dims(x, 0)
y_ = tf.placeholder(tf.float32, shape=[10])
W = tf.Variable(tf.zeros([784,10]))
T = tf.Variable(tf.zeros([10]))

#The Model:
#y = tf.nn.softmax(tf.matmul(x_reshape,W) + T)
y = tf.nn.sigmoid(tf.matmul(x_reshape,W) + T)

#Error Function
cost = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

#Set the Learning Rate = how quickly
LEARNING_RATE = 0.1

#define the learning function.
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

#initialise varialbe
init = tf.global_variables_initializer()
sess.run(init)

#some commands that will be used below for console output.
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

'''
for i in range(15):
    run_network_dd(i) 

input("Press [enter] to continue.")    
'''
 
for i in range(100):
    run_weight_ov(i)
#Import MNIST Dataset
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

'''
    The MNIST database is very large, so to speed things up we're only going to select a sample from the full database.
    
    The below functions will define a training and test dataset.
   
    This approach also allows us to experiment with the size of training set, and how it affects training and the end model.
'''

#Function that creates a training set of data, by taking the first 'num' from the MNIST training dataset
def TRAIN_SIZE(num):
    #The code
    x_train = mnist.train.images[:num,:]
    y_train = mnist.train.labels[:num,:]
    #just writing to the console
    print ('--------------------------------------------------')
    print ('x_train Examples Loaded = ' + str(x_train.shape))
    print ('y_train Examples Loaded = ' + str(y_train.shape))
    #return the dataset
    return x_train, y_train

print('\nLoading an example training set:')
#Use the above function to create a (very small) training set
x_train, y_train = TRAIN_SIZE(5)
   
#Function that creates a training set of data, by taking the first 'num' from the MNIST training dataset 
def TEST_SIZE(num):
    #The code
    x_test = mnist.test.images[:num,:]
    y_test = mnist.test.labels[:num,:]
    #just writing to the console
    print ('--------------------------------------------------')
    print ('x_test Examples Loaded = ' + str(x_test.shape))
    print ('y_test Examples Loaded = ' + str(y_test.shape))
    #return the dataset
    return x_test, y_test



'''
    We've set up some sets of data for us to use in training and testing. Now however lets actually have a look at the data.

    Each data point in this database consist of a 28x28 pixel picture of a handwritten digit (0-9), and a label denoting w. 
    
    Originally (NIST database) simply consisted of black and white pixels, they now contain greyscale due to the normalisation and anti-aliasing technique used. However even so this image can be represented as a 784 long array (list)
    
    The label consist of a 10 long array of 0's and 1, with the 1 deonting the correct digit.
    i.e. the label of '3' would be denoted by [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    
    In a neural network model, which array provides the inputs, and which array provides the target?
    
    The below functions will allow us to view a datapoint.
'''

#import some useful libraries - put this at the start of the code.
import matplotlib.pyplot as plt
import numpy as np
import random as ran

#function for printing a 784 array, in 28x28 format
def pretty_print_array(array, format="array"):
    """ This function will format an array with 784 elements into a 28 x 28 format
    By default it will present as an array (enclosed by square brackets)
    If the format is set to anything other than array it will not contain square brackets
    """
    n = 1
    pretty_string = ""

    if format == "array":
        pretty_string = pretty_string + "["

    for element in array:
        ## Every 28 elements we want to insert a new line
        if n % 28 == 0:
            ## The final element will not need a new line, but will need a closing square bracket if the format is an array
            if n == 784:
                pretty_string = pretty_string + str(element)
                if format == "array":
                    pretty_string = pretty_string + "]"
            else:
                pretty_string = pretty_string + str(element) + ",\n" 
        else:
            pretty_string = pretty_string + str(element) + ", "
        n += 1
    print(pretty_string)


#display the digit and the label
def display_train_digit(num):
    #turn the data points/arrays into strings then print them out
    #digit_array=str(x_train[num])
    digit_array=['%.2f' % elem for elem in x_train[num]]
    label_array=str(y_train[num])
    print('Label array is: ' + label_array)
    #print('Digit array is: ' + digit_array)
    pretty_print_array(digit_array)
    #plot the data points in a graph
    label = y_train[num].argmax(axis=0)
    image = x_train[num].reshape([28,28])
    plt.title('Label: %d' % (label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show()
    
def display_test_digit(num):
    ##turn the data points/arrays into strings then print them out
    label_array=str(y_test[num])
    label_list  = np.array(y_test[num]).tolist()
    print('Label Array: ' + label_array)
    print('Label: ' + str(label_list.index(max(label_list))))
    #plot the data points in a graph
    label = y_test[num].argmax(axis=0)
    image = x_test[num].reshape([28,28])
    plt.title('Label: %d' % (label))
    plt.imshow(image, cmap=plt.get_cmap('gray_r'))
    plt.show() 

print('\nDisplaying a random digit from the training dataset:')    
#This will display a random datapoint from the training dataset
display_train_digit(ran.randint(0, x_train.shape[0]-1))
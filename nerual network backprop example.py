# -*- coding: utf-8 -*-
# We will randomly define initial values for connection weights, and also randomly select
#   which training data that we will use for a given run.
import random

# We want to use the exp function (e to the x); it's part of our transfer function definition
from math import exp
import matplotlib.pyplot as plt

# Biting the bullet and starting to use NumPy for arrays
import numpy as np
from multiprocessing import pool, freeze_support
import itertools

# So we can make a separate list from an initial one
import copy

# For pretty-printing the arrays
np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


####################################################################################################
####################################################################################################
#
# Function to build up a trainingDataList
#
####################################################################################################
####################################################################################################

class Letter(object):
    """A class to hold one instance of a given letter.
    """
    def __init__(self, flattened_letter_array, target_letter):
        self.array = flattened_letter_array
        self.letter = target_letter
        self.target = ord(self.letter.lower()) - 97


def trainingSet():
    fullSet = []
    fullSet.append([1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 'X'])
    fullSet.append([1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 'M'])
    fullSet.append([1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 2, 'N'])
    fullSet.append([1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 3, 'H'])
    fullSet.append([0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 4, 'A'])
    fullSet.append([1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 5, 'I'])
    fullSet.append([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 6, 'L'])
    fullSet.append([1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 7, 'B'])
    return fullSet


def load_letters(size=25):
    """ Load all letters of given size (25, 81, etc)

    :param size:
    :return: number of unique letters and a list of Letter objects
    """
    from os import listdir
    basedir = 'U:\Midterm\letters'  # set this to match your environment!

    if size == 25:
        alpha_dir = basedir + '/5x5_alphabet/'
    elif size == 81:
        alpha_dir = basedir + '/9x9_alphabet/'

    letters = [f for f in listdir(alpha_dir)]
    letter_objects = []

    for letter in letters:
        letter_dir = alpha_dir + letter + '/'
        for instance in [f for f in listdir(letter_dir)]:
            instance_array = []
            with open(letter_dir + instance) as f:
                for line in f:
                    if '#' not in line:
                        instance_array.extend(line.replace(' ','').replace(',','').replace('\n',''))
            letter_objects.append(Letter([int(x) for x in instance_array],letter))

    return len(letters), letter_objects

def split_letters_into_train_and_test(letters, pct_test=20):
    """Split a given set of letters into training and test sets.

    :param letters:
    :param pct_test: a rough percentage of how many letters to hold back for testing
    :return: train, test
    """
    pass



####################################################################################################
####################################################################################################
#
# Procedure to print_welcome the user and identify the code
#
####################################################################################################
####################################################################################################


def print_welcome():
    print
    print '******************************************************************************'
    print
    print 'Welcome to the Multilayer Perceptron Neural Network'
    print '  trained using the backpropagation method.'
    print 'Version 0.2, 01/22/2017, A.J. Maren'
    print 'For comments, questions, or bug-fixes, contact: alianna.maren@northwestern.edu'
    print ' '
    print 'This program learns to distinguish between five capital letters: X, M, H, A, and N'
    print 'It allows users to examine the hidden weights to identify learned features'
    print
    print '******************************************************************************'
    print
    return ()


####################################################################################################
####################################################################################################
#
# A collection of worker-functions, designed to do specific small tasks
#
####################################################################################################
####################################################################################################


# ------------------------------------------------------#

# Compute neuron activation using sigmoid transfer function
def sigmoid_transfer(summedNeuronInput, alpha):
    return 1.0 / (1.0 + exp(-alpha * summedNeuronInput))


# ------------------------------------------------------#

# Compute derivative of transfer function
def backprop_transfer_derivative(NeuronOutput, alpha):
    return alpha * NeuronOutput * (1.0 - NeuronOutput)


# ------------------------------------------------------#
def dot_product(matrx1, matrx2):
    return np.dot(matrx1, matrx2)


def get_random_weight():
    return 1 - 2 * random.random()






def initialize_weights(dimensions):
    """Take in a dimension for an np.array and create an np.array, filled with random weights.
    :param dimensions: a list defining the dimensions of the array we wish to create.
    :return: a numpy array filled with random weights.
    """
    weights = np.zeros(dimensions)
    for weight in np.nditer(weights, op_flags=['readwrite']):
        weight[...] = get_random_weight()
    return weights


####################################################################################################
####################################################################################################
#
# Function to return a trainingDataList
#
####################################################################################################
####################################################################################################

def obtain_alphabet_training_value(trainingDataSetNum):
    return trainingSet()[trainingDataSetNum]


####################################################################################################
####################################################################################################
#
# Function to initialize a specific connection weight with a randomly-generated number between 0 & 1
#
####################################################################################################
####################################################################################################

def obtain_random_alphabet_training_values(num_training_datasets):
    # The training data list will have 11 values for the X-OR problem:
    #   - First 25 values will be the 5x5 pixel-grid representation of the letter
    #       represented as a 1-D array (0 or 1 for each)
    #   - Tenth value will be the output class (0 .. totalClasses - 1)
    #   - Eleventh value will be the string associated with that class, e.g., 'X'
    # We are starting with five letters in the training set: X, M, N, H, and A
    # Thus there are five choices for training data, which we'll select on random basis
    return obtain_alphabet_training_value(random.randint(0, num_training_datasets - 1))


def feed_forward(alpha, inputs, weights, biases):
    """Compute feed forward outputs

    :param alpha:
    :param inputs:
    :param weights:
    :param biases:
    :return: An array of output values from the hidden layer. Used as inputs to the output layer.
    """
    activations = dot_product(weights, inputs) + biases
    output = np.zeros(np.shape(activations))

    for node in range(len(output)):
        output[node] = sigmoid_transfer(activations[node], alpha)

    return output



def compute_outputs_and_errors(inputs, alpha, w_weights, hidden_layer_biases, v_weights, output_layer_biases,
                               print_results=False):
    """Compute the output node activations and determine errors across the entire training data set.

    :param alpha:
    :param w_weights:
    :param hidden_layer_biases:
    :param v_weights:
    :param output_layer_biases:
    :param print_results:
    :return:
    """
    errors = []
    selectedTrainingDataSet = 0
    inputArrayLength = w_weights.shape[1]
    hiddenArrayLength = w_weights.shape[0]
    outputArrayLength = v_weights.shape[0]

    for letter_object in inputs:
        letter_array = letter_object.array

        if print_results:
            print ' '
            print '  Data Set Number', selectedTrainingDataSet, ' for letter ', letter_object.letter

        hiddenArray = feed_forward(alpha, letter_array, w_weights, hidden_layer_biases)

        if print_results:
            print ' '
            print ' The hidden node activations are:'
            print hiddenArray

        outputArray = feed_forward(alpha, hiddenArray, v_weights, output_layer_biases)

        if print_results:
            print ' '
            print ' The output node activations are:'
            print outputArray

        target_output = np.zeros(outputArrayLength)  # initalize the output array with 0's
        target_out_node = letter_object.target  # identify the desired class
        target_output[target_out_node] = 1  # set the desired output for that class to 1

        if print_results:
            print ' '
            print ' The desired output array values are: '
            print target_output

        # Determine the error between actual and desired outputs

        # Initialize the error array
        errorArray = np.zeros(outputArrayLength)

        newSSE = 0.0
        for node in range(outputArrayLength):  # Number of nodes in output set (classes)
            errorArray[node] = target_output[node] - outputArray[node]
            newSSE = newSSE + errorArray[node] * errorArray[node]

        if print_results:
            print ' '
            print ' The error values are:'
            print errorArray

            # Print the Summed Squared Error
            print 'New SSE = %.6f' % newSSE
        errors.append(newSSE)

        selectedTrainingDataSet = selectedTrainingDataSet + 1

    return errors




####################################################################################################


# **************************************************************************************************#
####################################################################################################
#
#   Backpropgation Section
#
####################################################################################################
# **************************************************************************************************#
####################################################################################################



####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the hidden-to-output connection weights
#
####################################################################################################
####################################################################################################


def backprop_from_output_to_hidden(alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray, vWeightArray):
    # The first step here applies a backpropagation-based weight change to the hidden-to-output wts v.
    # Core equation for the first part of backpropagation:
    # d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
    # where:
    # -- SSE = sum of squared errors, and only the error associated with a given output node counts
    # -- v(h,o) is the connection weight v between the hidden node h and the output node o
    # -- alpha is the scaling term within the transfer function, often set to 1
    # ---- (this is included in transfFuncDeriv)
    # -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
    # -- F = transfer function, here using the sigmoid transfer function
    # -- Hidden(h) = the output of hidden node h.

    # We will DECREMENT the connection weight v by a small amount proportional to the derivative eqn
    #   of the SSE w/r/t the weight v.
    # This means, since there is a minus sign in that derivative, that we will add a small amount.
    # (Decrementing is -, applied to a (-), which yields a positive.)

    # For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand),
    #   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X.
    #   (Meaning: exact chapter is still TBD.)
    # For the latest updates, etc., please visit: www.aliannajmaren.com


    # Unpack array lengths
    hiddenArrayLength = arraySizeList[1]
    outputArrayLength = arraySizeList[2]

    transferFuncDerivArray = np.zeros(outputArrayLength)  # initalize an array for the transfer function

    for node in range(outputArrayLength):  # Number of hidden nodes
        transferFuncDerivArray[node] = backprop_transfer_derivative(outputArray[node], alpha)

    # Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
    #   and so is not included explicitly in the equations for the deltas in the connection weights
    #    print ' '
    #    print ' The transfer function derivative is: '
    #    print transferFuncDerivArray

    deltaVWtArray = np.zeros((outputArrayLength, hiddenArrayLength))  # initialize an array for the deltas
    newVWeightArray = np.zeros((outputArrayLength, hiddenArrayLength))  # initialize an array for the new hidden weights

    for row in range(outputArrayLength):  # Number of rows in weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes,
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input node array (expressed as a column).
        # Similarly, for a hidden-to-output matrix, the rows correspond to the number of output nodes,
        #    and the columns correspond to the number of hidden nodes,
        #    which can be multiplied by the hidden node array (expressed as a column).
        for col in range(hiddenArrayLength):  # number of columns in weightMatrix
            partialSSE_w_V_Wt = -errorArray[row] * transferFuncDerivArray[row] * hiddenArray[col]
            deltaVWtArray[row, col] = -eta * partialSSE_w_V_Wt
            newVWeightArray[row, col] = vWeightArray[row, col] + deltaVWtArray[row, col]

            #    print ' '
            #    print ' The previous hidden-to-output connection weights are: '
            #    print vWeightArray
            #    print ' '
            #    print ' The new hidden-to-output connection weights are: '
            #    print newVWeightArray

            #    PrintAndTraceBackpropagateOutputToHidden (alpha, nu, errorList, actualAllNodesOutputList,
            #    transFuncDerivList, deltaVWtArray, vWeightArray, newHiddenWeightArray)

    return newVWeightArray


####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the bias-to-output connection weights
#
####################################################################################################
####################################################################################################


def backprop_output_bias_weights(alpha, eta, arraySizeList, errorArray, outputArray, biasOutputWeightArray):
    # The first step here applies a backpropagation-based weight change to the hidden-to-output wts v.
    # Core equation for the first part of backpropagation:
    # d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
    # where:
    # -- SSE = sum of squared errors, and only the error associated with a given output node counts
    # -- v(h,o) is the connection weight v between the hidden node h and the output node o
    # -- alpha is the scaling term within the transfer function, often set to 1
    # ---- (this is included in transfFuncDeriv)
    # -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
    # -- F = transfer function, here using the sigmoid transfer function
    # -- Hidden(h) = the output of hidden node h.

    # Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n,
    #   scales amount of change to connection weight

    # We will DECREMENT the connection weight biasOutput by a small amount proportional to the derivative eqn
    #   of the SSE w/r/t the weight biasOutput(o).
    # This means, since there is a minus sign in that derivative, that we will add a small amount.
    # (Decrementing is -, applied to a (-), which yields a positive.)

    # For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand),
    #   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X.
    #   (Meaning: exact chapter is still TBD.)
    # For the latest updates, etc., please visit: www.aliannajmaren.com


    # Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
    #   and so is not included explicitly in these equations

    # The equation for the actual dependence of the Summed Squared Error on a given bias-to-output
    #   weight biasOutput(o) is:
    #   partial(SSE)/partial(biasOutput(o)) = -alpha*E(o)*F(o)*[1-F(o)]*1, as '1' is the input from the bias.
    # The transfer function derivative (transFuncDeriv) returned from backprop_transfer_derivative is given as:
    #   transFuncDeriv =  alpha*NeuronOutput*(1.0 -NeuronOutput), as with the hidden-to-output weights.
    # Therefore, we can write the equation for the partial(SSE)/partial(biasOutput(o)) as
    #   partial(SSE)/partial(biasOutput(o)) = E(o)*transFuncDeriv
    #   The parameter alpha is included in transFuncDeriv


    # Unpack the output array length
    outputArrayLength = arraySizeList[2]

    deltaBiasOutputArray = np.zeros(outputArrayLength)  # initialize an array for the deltas
    newBiasOutputWeightArray = np.zeros(outputArrayLength)  # initialize an array for the new output bias weights
    transferFuncDerivArray = np.zeros(outputArrayLength)  # initalize an array for the transfer function

    for node in range(outputArrayLength):  # Number of hidden nodes
        transferFuncDerivArray[node] = backprop_transfer_derivative(outputArray[node], alpha)

    for node in range(outputArrayLength):  # Number of nodes in output array (same as number of output bias nodes)
        partialSSE_w_BiasOutput = -errorArray[node] * transferFuncDerivArray[node]
        deltaBiasOutputArray[node] = -eta * partialSSE_w_BiasOutput
        newBiasOutputWeightArray[node] = biasOutputWeightArray[node] + deltaBiasOutputArray[node]

    # print ' '
    #    print ' The previous biases for the output nodes are: '
    #    print biasOutputWeightArray
    #    print ' '
    #    print ' The new biases for the output nodes are: '
    #    print newBiasOutputWeightArray

    return (newBiasOutputWeightArray);


####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the input-to-hidden connection weights
#
####################################################################################################
####################################################################################################


def backprop_from_hidden_to_input(alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray, inputArray,
                                  vWeightArray, wWeightArray):
    # The first step here applies a backpropagation-based weight change to the input-to-hidden wts w.
    # Core equation for the second part of backpropagation:
    # d(SSE)/dw(i,h) = -eta*alpha*F(h)(1-F(h))*Input(i)*sum(v(h,o)*Error(o))
    # where:
    # -- SSE = sum of squared errors, and only the error associated with a given output node counts
    # -- w(i,h) is the connection weight w between the input node i and the hidden node h
    # -- v(h,o) is the connection weight v between the hidden node h and the output node o
    # -- alpha is the scaling term within the transfer function, often set to 1
    # ---- (this is included in transfFuncDeriv)
    # -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
    # -- F = transfer function, here using the sigmoid transfer function
    # ---- NOTE: in this second step, the transfer function is applied to the output of the hidden node,
    # ------ so that F = F(h)
    # -- Hidden(h) = the output of hidden node h (used in computing the derivative of the transfer function).
    # -- Input(i) = the input at node i.

    # Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n,
    #   scales amount of change to connection weight

    # Unpack the errorList and the vWeightArray

    # We will DECREMENT the connection weight v by a small amount proportional to the derivative eqn
    #   of the SSE w/r/t the weight w.
    # This means, since there is a minus sign in that derivative, that we will add a small amount.
    # (Decrementing is -, applied to a (-), which yields a positive.)

    # For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand),
    #   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X.
    #   (Meaning: exact chapter is still TBD.)
    # For the latest updates, etc., please visit: www.aliannajmaren.com

    # Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n,
    #   scales amount of change to connection weight

    # For the second step in backpropagation (computing deltas on the input-to-hidden weights)
    #   we need the transfer function derivative is applied to the output at the hidden node

    # Unpack array lengths
    inputArrayLength = arraySizeList[0]
    hiddenArrayLength = arraySizeList[1]
    outputArrayLength = arraySizeList[2]

    # Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
    #   and so is not included explicitly in these equations
    transferFuncDerivHiddenArray = np.zeros(hiddenArrayLength)  # initialize an array for the transfer function deriv

    for node in range(hiddenArrayLength):  # Number of hidden nodes
        transferFuncDerivHiddenArray[node] = backprop_transfer_derivative(hiddenArray[node], alpha)

    errorTimesTFuncDerivOutputArray = np.zeros(outputArrayLength)  # initialize array
    transferFuncDerivOutputArray = np.zeros(outputArrayLength)  # initialize array
    weightedErrorArray = np.zeros(hiddenArrayLength)  # initialize array

    for outputNode in range(outputArrayLength):  # Number of output nodes
        transferFuncDerivOutputArray[outputNode] = backprop_transfer_derivative(outputArray[outputNode], alpha)
        errorTimesTFuncDerivOutputArray[outputNode] = errorArray[outputNode] * transferFuncDerivOutputArray[outputNode]

    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  # Number of output nodes
            weightedErrorArray[hiddenNode] = weightedErrorArray[hiddenNode] \
                                             + vWeightArray[outputNode, hiddenNode] * errorTimesTFuncDerivOutputArray[
                outputNode]

    deltaWWtArray = np.zeros((hiddenArrayLength, inputArrayLength))  # initialize an array for the deltas
    newWWeightArray = np.zeros(
        (hiddenArrayLength, inputArrayLength))  # initialize an array for the new input-to-hidden weights

    for row in range(hiddenArrayLength):  # Number of rows in input-to-hidden weightMatrix
        # For an input-to-hidden weight matrix, the rows correspond to the number of hidden nodes,
        #    and the columns correspond to the number of input nodes.
        #    This creates an HxI matrix, which can be multiplied by the input node array (expressed as a column).

        for col in range(inputArrayLength):  # number of columns in weightMatrix
            partialSSE_w_W_Wts = -transferFuncDerivHiddenArray[row] * inputArray[col] * weightedErrorArray[row]
            deltaWWtArray[row, col] = -eta * partialSSE_w_W_Wts
            newWWeightArray[row, col] = wWeightArray[row, col] + deltaWWtArray[row, col]

            #    print ' '
            #    print ' The previous hidden-to-output connection weights are: '
            #    print wWeightArray
            #    print ' '
            #    print ' The new hidden-to-output connection weights are: '
            #    print newWWeightArray

    return newWWeightArray


####################################################################################################
####################################################################################################
#
# Backpropagate weight changes onto the bias-to-hidden connection weights
#
####################################################################################################
####################################################################################################


def backprop_hidden_bias_weights(alpha, eta, arraySizeList, errorArray, outputArray, hiddenArray, vWeightArray,
                                 biasHiddenWeightArray):
    # The first step here applies a backpropagation-based weight change to the hidden-to-output wts v.
    # Core equation for the first part of backpropagation:
    # d(SSE)/dv(h,o) = -alpha*Error*F(1-F)*Hidden(h)
    # where:
    # -- SSE = sum of squared errors, and only the error associated with a given output node counts
    # -- v(h,o) is the connection weight v between the hidden node h and the output node o
    # -- alpha is the scaling term within the transfer function, often set to 1
    # ---- (this is included in transfFuncDeriv)
    # -- Error = Error(o) or error at the output node o; = Desired(o) - Actual(o)
    # -- F = transfer function, here using the sigmoid transfer function
    # -- Hidden(h) = the output of hidden node h.

    # Note that the training rate parameter is assigned in main; Greek letter "eta," looks like n,
    #   scales amount of change to connection weight

    # We will DECREMENT the connection weight biasOutput by a small amount proportional to the derivative eqn
    #   of the SSE w/r/t the weight biasOutput(o).
    # This means, since there is a minus sign in that derivative, that we will add a small amount.
    # (Decrementing is -, applied to a (-), which yields a positive.)

    # For the actual derivation of this equation with MATCHING VARIABLE NAMES (easy to understand),
    #   please consult: Brain-Based Computing, by AJ Maren (under development, Jan., 2017). Chpt. X.
    #   (Meaning: exact chapter is still TBD.)
    # For the latest updates, etc., please visit: www.aliannajmaren.com


    # Unpack array lengths
    inputArrayLength = arraySizeList[0]
    hiddenArrayLength = arraySizeList[1]
    outputArrayLength = arraySizeList[2]

    # Compute the transfer function derivatives as a function of the output nodes.
    # Note: As this is being done after the call to the backpropagation on the hidden-to-output weights,
    #   the transfer function derivative computed there could have been used here; the calculations are
    #   being redone here only to maintain module independence

    errorTimesTFuncDerivOutputArray = np.zeros(outputArrayLength)  # initialize array
    transferFuncDerivOutputArray = np.zeros(outputArrayLength)  # initialize array
    weightedErrorArray = np.zeros(hiddenArrayLength)  # initialize array

    transferFuncDerivHiddenArray = np.zeros(hiddenArrayLength)  # initialize an array for the transfer function deriv
    partialSSE_w_BiasHidden = np.zeros(hiddenArrayLength)  # initialize an array for the partial derivative of the SSE
    deltaBiasHiddenArray = np.zeros(hiddenArrayLength)  # initialize an array for the deltas
    newBiasHiddenWeightArray = np.zeros(hiddenArrayLength)  # initialize an array for the new hidden bias weights

    for node in range(hiddenArrayLength):  # Number of hidden nodes
        transferFuncDerivHiddenArray[node] = backprop_transfer_derivative(hiddenArray[node], alpha)

    for outputNode in range(outputArrayLength):  # Number of output nodes
        transferFuncDerivOutputArray[outputNode] = backprop_transfer_derivative(outputArray[outputNode], alpha)
        errorTimesTFuncDerivOutputArray[outputNode] = errorArray[outputNode] * transferFuncDerivOutputArray[outputNode]

    for hiddenNode in range(hiddenArrayLength):
        weightedErrorArray[hiddenNode] = 0
        for outputNode in range(outputArrayLength):  # Number of output nodes
            weightedErrorArray[hiddenNode] = weightedErrorArray[hiddenNode]
            + vWeightArray[outputNode, hiddenNode] * errorTimesTFuncDerivOutputArray[outputNode]

            # Note: the parameter 'alpha' in the transfer function shows up in the transfer function derivative
            #   and so is not included explicitly in these equations


            # ===>>> AJM needs to double-check these equations in the comments area
            # ===>>> The code should be fine.
            # The equation for the actual dependence of the Summed Squared Error on a given bias-to-output
            #   weight biasOutput(o) is:
            #   partial(SSE)/partial(biasOutput(o)) = -alpha*E(o)*F(o)*[1-F(o)]*1, as '1' is the input from the bias.
            # The transfer function derivative (transFuncDeriv) returned from backprop_transfer_derivative is given as:
            #   transFuncDeriv =  alpha*NeuronOutput*(1.0 -NeuronOutput), as with the hidden-to-output weights.
            # Therefore, we can write the equation for the partial(SSE)/partial(biasOutput(o)) as
            #   partial(SSE)/partial(biasOutput(o)) = E(o)*transFuncDeriv
            #   The parameter alpha is included in transFuncDeriv

    for hiddenNode in range(hiddenArrayLength):  # Number of rows in input-to-hidden weightMatrix
        partialSSE_w_BiasHidden[hiddenNode] = -transferFuncDerivHiddenArray[hiddenNode] * weightedErrorArray[hiddenNode]
        deltaBiasHiddenArray[hiddenNode] = -eta * partialSSE_w_BiasHidden[hiddenNode]
        newBiasHiddenWeightArray[hiddenNode] = biasHiddenWeightArray[hiddenNode] + deltaBiasHiddenArray[hiddenNode]

    return (newBiasHiddenWeightArray);

####################################################################################################
####################################################################################################
#
# Check if errors for all letters are below Epsilon
#
####################################################################################################
####################################################################################################

def all_errors_are_below_epsilon(errors, epsilon):
    for error in errors:
        if error > epsilon:
            return False

    #got here and no errors > epsilon
    return True

####################################################################################################
####################################################################################################
#
# compute sum of squared errors for a specific letter
#
####################################################################################################
####################################################################################################

def get_sse(desiredOutputArray, errorArray, outputArray, outputArrayLength):
    newSSE = 0.0
    for node in range(outputArrayLength):  # Number of nodes in output set (classes)
        errorArray[node] = desiredOutputArray[node] - outputArray[node]
        newSSE = newSSE + errorArray[node] * errorArray[node]
    return newSSE


def run_neural_network(num_hidden_nodes, alpha, epsilon, eta, max_iterations, letter_size):
    iteration = 0
    numTrainingDataSets = len(trainingSet())
    SSE = 0.0

    num_of_letters, all_letters = load_letters(letter_size)
    input_nodes = len(all_letters[0].array)
    output_nodes = num_of_letters

    network_dimensions = [input_nodes, num_hidden_nodes, output_nodes]


    ####################################################################################################
    # Initialize the weight arrays for two sets of weights; w: input-to-hidden, and v: hidden-to-output
    ####################################################################################################
    #
    # The w_weights is for Input-to-Hidden
    # The v_weights is for Hidden-to-Output
    w_weight_dimensions = [num_hidden_nodes, input_nodes]
    v_weight_dimensions = [output_nodes, num_hidden_nodes]
    hidden_layer_bias_nodes = num_hidden_nodes
    output_layer_bias_nodes = output_nodes
    # The node-to-node connection weights are stored in a 2-D array
    w_weights = initialize_weights(w_weight_dimensions)
    v_weights = initialize_weights(v_weight_dimensions)
    # The bias weights are stored in a 1-D array
    hidden_biases = initialize_weights(hidden_layer_bias_nodes)
    output_biases = initialize_weights(output_layer_bias_nodes)
    # Notice in the very beginning of the program, we have
    #   np.set_printoptions(precision=4) (sets number of dec. places in print)
    #     and 'np.set_printoptions(suppress=True)', which keeps it from printing in scientific format
    #   Debug print:
    #    print
    #    print 'The initial weights for this neural network are:'
    #    print '       Input-to-Hidden '
    #    print w_weights
    #    print '       Hidden-to-Output'
    #    print v_weights
    #    print ' '
    #    print 'The initial bias weights for this neural network are:'
    #    print '        Hidden Bias = ', hidden_biases
    #    print '        Output Bias = ', output_biases
    ####################################################################################################
    # Before we start training, get a baseline set of outputs, errors, and SSE
    ####################################################################################################
    print ' '
    print '  Before training:'
    compute_outputs_and_errors(all_letters, alpha, w_weights, hidden_biases, v_weights, output_biases, print_results=True)
    ####################################################################################################
    # Next step - Obtain a single set of randomly-selected training values for alpha-classification
    ####################################################################################################
    for iteration in range(max_iterations):
        random_letter = random.choice(all_letters)
        inputDataList = random_letter.array
        desiredOutputArray = np.zeros(output_nodes)  # initalize the output array with 0's
        desiredClass = random_letter.target # identify the desired class
        desiredOutputArray[desiredClass] = 1  # set the desired output for that class to 1

        #        print ' '
        #        print ' The desired output array values are: '
        #        print desiredOutputArray
        #        print ' '



        ####################################################################################################
        # Compute a single feed-forward pass and obtain the Actual Outputs
        ####################################################################################################
        hiddenArray = feed_forward(alpha, inputDataList, w_weights, hidden_biases)
        outputArray = feed_forward(alpha, hiddenArray, v_weights, output_biases)

        errorArray = np.zeros(output_nodes)
        newSSE = 0.0
        for node in range(output_nodes):  # Number of nodes in output set (classes)
            errorArray[node] = desiredOutputArray[node] - outputArray[node]
            newSSE = newSSE + errorArray[node] * errorArray[node]

        # print ' '
        #        print ' The error values are:'
        #        print errorArray

        # Print the Summed Squared Error
        #        print 'Initial SSE = %.6f' % newSSE
        #        SSE = newSSE



        ####################################################################################################
        # Perform backpropagation
        ####################################################################################################


        # Perform first part of the backpropagation of weight changes
        newVWeightArray = backprop_from_output_to_hidden(alpha, eta, network_dimensions, errorArray, outputArray,
                                                         hiddenArray, v_weights)
        newBiasOutputWeightArray = backprop_output_bias_weights(alpha, eta, network_dimensions, errorArray, outputArray,
                                                                output_biases)

        # Perform first part of the backpropagation of weight changes
        newWWeightArray = backprop_from_hidden_to_input(alpha, eta, network_dimensions, errorArray, outputArray,
                                                        hiddenArray, inputDataList, v_weights, w_weights)

        newBiasHiddenWeightArray = backprop_hidden_bias_weights(alpha, eta, network_dimensions, errorArray, outputArray,
                                                                hiddenArray, v_weights, hidden_biases)

        # Assign new values to the weight matrices
        # Assign the old hidden-to-output weight array to be the same as what was returned from the BP weight update
        v_weights = newVWeightArray[:]

        output_biases = newBiasOutputWeightArray[:]

        # Assign the old input-to-hidden weight array to be the same as what was returned from the BP weight update
        w_weights = newWWeightArray[:]

        hidden_biases = newBiasHiddenWeightArray[:]

        # Compute a forward pass, test the new SSE

        hiddenArray = feed_forward(alpha, inputDataList, w_weights, hidden_biases)

        #    print ' '
        #    print ' The hidden node activations are:'
        #    print hiddenArray

        outputArray = feed_forward(alpha, hiddenArray, v_weights, output_biases)

        #    print ' '
        #    print ' The output node activations are:'
        #    print outputArray


        # Determine the error between actual and desired outputs

        newSSE = get_sse(desiredOutputArray, errorArray, outputArray, output_nodes)  # print ' '
        #        print ' The error values are:'
        #        print errorArray

        # Print the Summed Squared Error
        #        print 'Previous SSE = %.6f' % SSE
        #        print 'New SSE = %.6f' % newSSE

        #        print ' '
        #        print 'Iteration number ', iteration
        #        iteration = iteration + 1

        if newSSE < epsilon:
            errors = compute_outputs_and_errors(all_letters, alpha, w_weights, hidden_biases,
                                                v_weights, output_biases, print_results=False)

            if all_errors_are_below_epsilon(errors, epsilon):
                break

    print 'Out of while loop at iteration ', iteration
    ####################################################################################################
    # After training, get a new comparative set of outputs, errors, and SSE
    ####################################################################################################
    print ' '
    print '  After training:'
    compute_outputs_and_errors(all_letters, alpha, w_weights, hidden_biases,
                               v_weights, output_biases, print_results=True)

    result = {}
    result["num_hidden_nodes"] = num_hidden_nodes
    result["iterations"] = iteration
    result["alpha"] = alpha
    result["eta"] = eta
    return result

def neural_network_wrapper_for_multiprocessing(params):
    return run_neural_network(*params)

def plot_results(results):
    alphas = []
    etas = []
    iterations_in_each_result = []
    for result in results:
        iterations_in_each_result.append(result["iterations"])
        alphas.append(result["alpha"])
        etas.append(result["eta"])

    plt.scatter(alphas, etas, s=iterations_in_each_result)
    plt.title("Iterations to Convergence")
    plt.xlabel('Sigmoid Transfer (alpha)')
    plt.ylabel('Learning Rate (eta)')
    plt.show()


####################################################################################################
# **************************************************************************************************#
####################################################################################################
#
# The MAIN module comprising of calls to:
#   (1) Welcome
#   (2) Obtain neural network size specifications for a three-layer network consisting of:
#       - Input layer
#       - Hidden layer
#       - Output layer (all the sizes are currently hard-coded to two nodes per layer right now)
#   (3) Initialize connection weight values
#       - w: Input-to-Hidden nodes
#       - v: Hidden-to-Output nodes
#   (4) Compute a feedforward pass in two steps
#       - Randomly select a single training data set
#       - Input-to-Hidden
#       - Hidden-to-Output
#       - Compute the error array
#       - Compute the new Summed Squared Error (SSE)
#   (5) Perform a single backpropagation training pass
#
####################################################################################################
# **************************************************************************************************#
####################################################################################################


def main():
    ####################################################################################################
    # Obtain unit array size in terms of array_length (M) and layers (N)
    ####################################################################################################

    # This calls the procedure 'print_welcome,' which just prints out a welcoming message.
    # All procedures need an argument list.
    # This procedure has a list, but it is an empty list; print_welcome().

    print_welcome()


    # Parameter definitions, to be replaced with user inputs
    max_iterations = 10000  # temporarily set to 10 for testing
    epsilon = 0.05

    # removed 0.25 because it never converged
    alphas_to_try = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    etas_to_try = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
    hidden_nodes_to_try = [5, 10, 15, 20]
    results = []
    pool_size = 8
    p = pool.Pool(pool_size)

    for numHiddenNodes in hidden_nodes_to_try:
        for alpha in alphas_to_try:
            etaResults = p.map(neural_network_wrapper_for_multiprocessing,
                               itertools.izip(itertools.repeat(numHiddenNodes),
                                              itertools.repeat(alpha),
                                              itertools.repeat(epsilon),
                                              etas_to_try,
                                              itertools.repeat(max_iterations),
                                              [81]
                                              ))
            for etaResult in etaResults:
                results.append(etaResult)

        #  Code to run serial
        # for eta in etas_to_try:
        #     result = {}
        #     result["iterations"] = run_neural_network(alpha, epsilon, eta, max_iterations)
        #     result["alpha"] = alpha
        #     result["eta"] = eta
        #     results.append(result)

    for result in results:
        print(result)
    plot_results(results)


####################################################################################################
# Conclude specification of the MAIN procedure
####################################################################################################

if __name__ == "__main__":
    main()



####################################################################################################
####################################################################################################
#
# Function to obtain a randomly-selected training data set list, which will contain:
#   First element: The number of the training data set (put in a placeholder number, AJM will re-assign),
#   Second element: An 81-element 1-D binary array, which gives the pattern of your alphabet value, 
#   values (each either 0 or 1), and two output values (see list below), and a third value which is 
#   the number of the training data set, in the range of (0..3). (There are a total of four training 
#   data sets.) 
#
####################################################################################################
####################################################################################################

def obtainRandomAlphabetTrainingValues ():
    
# JUST STICK YOUR DATA SET IN AFTER MINE, please & thank you, upload when done
# If you have multiple data sets (different versions of the same letter), please put them all in. 
# One line per each. Thx! - AJM    

   
    # The training data list will have four values for the X-OR problem:
    #   - First two valuea will be the two inputs (0 or 1 for each)
    #   - Second two values will be the two outputs (0 or 1 for each)
    # There are only four combinations of 0 and 1 for the input data
    # Thus there are four choices for training data, which we'll select on random basis
    
    # The fifth element in the list is the NUMBER of the training set; setNumber = 1..3
    # The setNumber is used to assign the Summed Squared Error (SSE) with the appropriate
    #   training set
    # This is because we need ALL the SSE's to get below a certain minimum before we stop
    #   training the network

    # Right now, just put down your training set, in the form similar to mine. 
    # I'll go back and put some code in to return a randomly-selected training data set. 
            
    #trainingDataListN = (14,[1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,1,0,1,0,0,0,0,0,1,1,0,0,1,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,0,0,1,1,0,0,0,0,0,1,0,1,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1],14,'N') # training data list 14 selected for the letter 'N'
    #trainingDataListA = (1,[0,0,0,0,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1],1,'A') # training data list 14 selected for the letter 'N'

    #From Dong Bing for the letter Z
    trainingDataListZ = (26,[0,1,1,1,1,1,1,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,1,0,0, 0,0,0,0,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0,0, 0,1,0,0,0,0,0,0,0, 0,1,1,1,1,1,1,1,0],26,'Z') # Dong Bing training data list 14 selected for the letter 'Z'


    return (trainingDataList)  


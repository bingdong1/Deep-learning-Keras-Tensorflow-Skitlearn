# -*- coding: utf-8 -*-

from random import randint

####################################################################################################
####################################################################################################
#
# Function to obtain a randomly-selected training data set list, which will contain:
#   First element: The number of the training data set (put in a placeholder number, AJM will re-assign)
#    (Note: There will likely be over four dozen training data sets, accounting for variances), 
#   Second element: An 81-element 1-D binary array, which gives the pattern of your alphabet value, 
#     values (each either 0 or 1), 
#   Third element: the number of the training data set output, in the range of (0.. 26),
#   Fourth element: the 'string' version of the output, e.g., 'A' is associated with the first desired output. 
#   Fifth element: the number of the 'big letter class' with which the letter is associated
#   Sixth element: the most characteristic letter for the 'big letter class'
#
####################################################################################################
####################################################################################################

def obtainAlphabetTrainingValues (dataSet):
    
# Various training data sets for the alphabet challenge. There are frequently more than one version of the 
#   same letter. The primary version of each letter is given in training data sets A..Z, then the variants are 
#   stored in approximately alphabetic order.     

    trainingDataListA0  =   (1,[0,0,0,0,1,0,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,1,0,0,0,1,0,0, 0,1,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],1,'A',1,'A') # training data list 1 selected for the letter 'A'
    trainingDataListB0  =   (2,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,0],2,'B',2,'B') # training data list 2, letter 'E', courtesy AJM
    trainingDataListC0  =   (3,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],3,'C',3,'C') # training data list 3, letter 'C', courtesy PKVR
    trainingDataListD0  =   (4,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1,1,0],4,'D',4,'O') # training data list 4, letter 'D', courtesy TD
    trainingDataListE0  =   (5,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],5,'E',5,'E') # training data list 5, letter 'E', courtesy BMcD 
    trainingDataListF0  =   (6,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],6,'F',5,'E') # training data list 6, letter 'F', courtesy SK
    trainingDataListG0  =   (7,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1],7,'G',3,'C') # training data list 7, letter 'G', courtesy AJM
    trainingDataListH0  =   (8,[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],8,'H',1,'A') # training data list 8, letter 'H', courtesy JC
    trainingDataListI0  =   (9,[0,0,1,1,1,1,1,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,1,1,1,1,1,0,0],9,'I',6,'I') # training data list 9, letter 'I', courtesy GR
    trainingDataListJ0  =  (10,[0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,1,1,0,0,0],10,'J',6,'I') # training data list 10 selected for the letter 'L', courtesy JT
    trainingDataListK0  =  (11,[1,0,0,0,0,0,1,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,1,1,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,0,1,0,0],11,'K',7,'K') # training data list 11 selected for the letter 'K', courtesy EO      
    trainingDataListL0  =  (12,[1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],12,'L',8,'L') # training data list 12 selected for the letter 'L', courtesy PV
    trainingDataListM0  =  (13,[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,0,1,0,0,0,1,0,1, 1,0,1,0,0,0,1,0,1, 1,0,0,1,0,1,0,0,1, 1,0,0,1,0,1,0,0,1, 1,1,0,0,1,0,0,0,1, 1,0,0,0,1,0,0,0,1],13,'M',9,'M') # training data list 13 selected for the letter 'M', courtesy GR            
    trainingDataListN0  =  (14,[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,0,1, 1,0,1,0,0,0,0,0,1, 1,0,0,1,0,0,0,0,1, 1,0,0,0,1,0,0,0,1, 1,0,0,0,0,1,0,0,1, 1,0,0,0,0,0,1,0,1, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1],14,'N',9,'M') # training data list 14 selected for the letter 'N'
    trainingDataListO0  =  (15,[0,1,1,1,1,1,1,1,0, 1,1,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 0,1,1,1,1,1,1,1,0],15,'O',4,'O') # training data list 15, letter 'O', courtesy TD
    trainingDataListP0  =  (16,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],15,'P',1,'B') # training data list 16, letter 'P', courtesy MT 
    trainingDataListQ0  =  (17,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,1,0,0,1, 1,0,0,0,0,0,1,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1,1,1],17,'Q',3,'O') # training data list 17, letter 'Q', courtesy AJM (square corners)
    trainingDataListR0  =  (18,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,0,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1],15,'R',1,'B') # training data list 18, letter 'R', courtesy AJM (variant on 'P') 
    trainingDataListS0  =  (19,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1],19,'S',5,'E') # training data list 19, letter 'S', courtesy RG (square corners)
    trainingDataListT0  =  (20,[0,1,1,1,1,1,1,1,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0],20,'T',6,'I') # training data list 20, letter 'T', courtesy JR
    trainingDataListU0  =  (21,[1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 0,1,1,0,0,0,1,1,0, 0,0,1,1,1,1,1,0,0],21,'U',8,'L') # training data list 21, letter 'U', courtesy JD
    trainingDataListV0  =  (22,[1,0,0,0,0,0,0,0,1, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,1,1,0,1,1,0,0, 0,0,0,1,0,1,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0],22,'V',9,'V') # training data list 22, letter 'V', courtesy JW, Cohort 3 
    trainingDataListW0  =  (23,[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,1,0,0,0,1, 1,0,0,1,0,1,0,0,1, 1,0,1,0,0,0,1,0,1, 0,1,0,0,0,0,0,1,0],23,'W',10,'W') # training data list 23, letter 'W', courtesy KW, Cohort 1
    trainingDataListX0  =  (24,[1,0,0,0,0,0,0,0,1, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,1,0,0,0,1,0,0, 0,1,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1],24,'X',11,'X') # training data list 24, letter 'X', courtesy JD, Cohort 1                     
    trainingDataListY0  =  (25,[1,0,0,0,0,0,0,0,1, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0],25,'Y',11,'X') # training data list 25, letter 'Y', courtesy ZCP, Cohort 1  
    trainingDataListZ0  =  (26,[1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,1,0,0, 0,0,0,0,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0,0, 0,1,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],26,'Z',11,'X') # training data list 26, letter 'Z', courtesy ZW, Cohort 1

                                                                                                

# Training data with rounded (or more rounded) sides (numbering from 101 ++)
    trainingDataListAr1 = (101,[0,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],1,'A', 1,'A') # training data list 101, letter 'A', var1, courtesy JBA, Cohort 3; rounded upper corners
    trainingDataListPr1 = (116,[1,1,1,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,1,1,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],16,'P',1,'B')  # training data list 116, letter 'P', var1, courtesy PH, Cohort 3; rounded shape
    trainingDataListYr1 = (125,[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 0,1,0,0,0,0,0,1,0, 0,0,1,1,1,1,1,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0],25,'Y',11,'X') # training data list 125, letter 'Y', var1, courtesy ZCP Cohort 1; rounded sides 



# Training data with serifs (numbering from 201 ++)
    trainingDataListHs1 = (208,[1,1,1,0,0,1,1,1,0, 0,1,0,0,0,0,1,0,0, 0,1,0,0,0,0,1,0,0, 0,1,0,0,0,0,1,0,0, 0,1,1,1,1,1,1,0,0, 0,1,0,0,0,0,1,0,0, 0,1,0,0,0,0,1,0,0, 0,1,0,0,0,0,1,0,0, 1,1,1,0,0,1,1,1,0],8,'H',1,'A')   # training data list 208, letter 'H', var2, courtesy Joe C, Cohort 3; serif
    trainingDataListHs2 = (238,[1,1,1,0,0,0,1,1,1, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,1,1,1,1,1,1,1,0, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 1,1,1,0,0,0,1,1,1],8,'H',1,'A')   # training data list 238, letter 'H', var3, courtesy Joe C, Cohort 3; serif

    trainingDataListJs1 = (310,[0,0,1,1,1,1,1,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,1,0,0,1,0,0,0,0, 0,1,0,0,1,0,0,0,0, 0,1,1,1,1,0,0,0,0],10,'J',6,'I')  # training data list 210, letter 'J', var1, courtesy JS, Cohort 3; serif 

    trainingDataListWs1 = (223,[1,1,1,0,0,0,1,1,1, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,1,0,0,1,0,0,1,0, 0,1,0,1,0,1,0,1,0, 0,1,1,0,0,0,1,1,0, 0,1,0,0,0,0,0,1,0],23,'W',10,'M') # training data list 223, letter 'W', var1, courtesy CG, Cohort 2; serif, compressed
    trainingDataListYs1 = (225,[1,1,1,0,0,0,1,1,1, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,1,1,1,0,0,0],25,'Y',11,'X') # training data list 225, letter 'Y', var1, courtesy ZCP Cohort 1; serif, compressed      


# Training data with translated (up/down) or shifted (e.g., w/ letter 'J') or squeezed (left/right, e.g., letter 'H') line segments (numbering from 301 ++)

    trainingDataListPt1 = (316,[1,1,1,1,1,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,1,1,1,1,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],16,'P',1,'B')  # training data list 316, letter 'P', var2, courtesy PH, Cohort 3; compressed
    trainingDataListPt2 = (346,[1,1,1,1,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,1,1,1,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],16,'P',1,'B')  # training data list 346, letter 'P', var3, courtesy PH, Cohort 3; compressed

    trainingDataListHt1 = (308,[0,0,1,0,0,0,1,0,0, 0,0,1,0,0,0,1,0,0, 0,0,1,0,0,0,1,0,0, 0,0,1,0,0,0,1,0,0, 0,0,1,1,1,1,1,0,0, 0,0,1,0,0,0,1,0,0, 0,0,1,0,0,0,1,0,0, 0,0,1,0,0,0,1,0,0, 0,0,1,0,0,0,1,0,0],8,'H',1,'A') # training data list 308, letter 'H', var1, courtesy Joe C, Cohort 3; compressed



    trainingDataListKt1 = (311,[1,0,0,0,0,1,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,1,0,0,0,0,0,0, 1,1,0,0,0,0,0,0,0, 1,0,1,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,0,0,1,0,0,0],11,'K',7,'K') # training data list 311, letter 'K', var1, courtesy MTB, Cohort 3; compressed/shifted
    trainingDataListKt2 = (341,[1,0,0,0,1,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,1,0,0,0,0,0,0, 1,1,0,0,0,0,0,0,0, 1,1,0,0,0,0,0,0,0, 1,0,1,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,0,0,1,0,0,0],11,'K',7,'K') # training data list 341, letter 'K', var2, courtesy MTB, Cohort 3; compressed/shifted
    trainingDataListKt3 = (371,[1,0,0,0,1,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,1,0,0,0,0,0,0, 1,1,0,1,0,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,1,0,0,0],11,'K',7,'K') # training data list 371, letter 'K', var3, courtesy MTB, Cohort 3; compressed/shifted
    trainingDataListKt4 = (372,[0,0,1,0,0,0,1,0,0, 0,0,1,0,0,1,0,0,0, 0,0,1,0,1,0,0,0,0, 0,0,1,1,0,0,0,0,0, 0,0,1,1,0,0,0,0,0, 0,0,1,0,1,0,0,0,0, 0,0,1,0,0,1,0,0,0, 0,0,1,0,0,0,1,0,0, 0,0,1,0,0,0,0,1,0],11,'K',7,'K') # training data list 372, letter 'K', var4, courtesy MTB, Cohort 3; compressed/shifted
    trainingDataListKt5 = (373,[0,0,1,0,0,0,1,0,0, 0,0,1,0,0,0,1,0,0, 0,0,1,0,0,1,0,0,0, 0,0,1,0,1,0,0,0,0, 0,0,1,1,0,1,0,0,0, 0,0,1,0,0,0,1,0,0, 0,0,1,0,0,0,1,0,0, 0,0,1,0,0,0,0,1,0, 0,0,1,0,0,0,0,1,0],11,'K',7,'K') # training data list 373, letter 'K', var5, courtesy MTB, Cohort 3; compressed/shifted
    trainingDataListKt6 = (374,[0,1,0,0,0,1,0,0,0, 0,1,0,0,0,1,0,0,0, 0,1,0,0,1,0,0,0,0, 0,1,0,0,1,0,0,0,0, 0,1,1,1,0,0,0,0,0, 0,1,0,0,1,0,0,0,0, 0,1,0,0,0,1,0,0,0, 0,1,0,0,0,1,0,0,0, 0,1,0,0,0,1,0,0,0],11,'K',7,'K') # training data list 374, letter 'K', var6, courtesy MTB, Cohort 3; compressed/shifted       

    trainingDataListLt1 = (312,[1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,0,0,0,0],12,'L',8,'L') # training data list 312, letter 'L', var1, courtesy DS, Cohort 3; compressed
    trainingDataListLt2 = (342,[1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,0,0,0],12,'L',8,'L') # training data list 342, letter 'L', var1, courtesy DS, Cohort 3; compressed
    
    
    
    trainingDataListYt1 = (325,[0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0],25,'Y',11,'X') # training data list 325, letter 'Y', var1, courtesy ZCP, Cohort 1; compressed       
    trainingDataListZt1 = (326,[0,1,1,1,1,1,1,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,1,0,0, 0,0,0,0,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0,0, 0,1,0,0,0,0,0,0,0, 0,1,1,1,1,1,1,1,0],26,'Z',11,'X') # training data list 326, letter 'Z', var1, courtesy DB, Cohort 3; compressed   

# Training data with proportions changed (up/down on vertices, e.g. w/ letters 'Y' or 'W')  (numbering from 401 ++)

 
    trainingDataListFx1 = (406,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0], 6, 'F',5,'E')  # training data list 406, letter 'F'; var1, courtesy SWA, Cohort 3; middle bar one level lower 
    trainingDataListFx2 = (436,[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0], 6, 'F',5,'E')  # training data list 436, letter 'F'; var2, courtesy SWA, Cohort 3; half of middle bar 



# Training data with affine transformations (e.g., tilted) (numbering from 501 ++)

    trainingDataListAa1 = (501,[1,0,0,0,0,0,0,0,0, 1,1,0,0,0,0,0,0,0, 1,0,1,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,1,1,1,1,1,0,0,0, 1,0,0,0,0,0,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1],1,'A', 1,'A') # training data list 501, letter 'A', var3, courtesy JBA, Cohort 3; strong tilt to left (affine transform)
    trainingDataListAa2 = (531,[0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,1,1, 0,0,0,0,0,0,1,0,1, 0,0,0,0,0,1,0,0,1, 0,0,0,0,1,0,0,0,1, 0,0,0,1,1,1,1,1,1, 0,0,1,0,0,0,1,0,0, 0,1,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1],1,'A', 1,'A') # training data list 531, letter 'A', var4, courtesy JBA, Cohort 3; strong tilt to right (affine transform)


# Training data with double-thickness strokes (numbering from 601 ++)

    trainingDataListJd1 = (610,[0,0,1,1,1,1,1,1,0, 0,0,1,1,1,1,1,1,0, 0,0,0,0,1,1,0,0,0, 0,0,0,0,1,1,0,0,0, 0,0,0,0,1,1,0,0,0, 1,1,0,0,1,1,0,0,0, 1,1,0,0,1,1,0,0,0, 1,1,1,1,1,1,0,0,0, 1,1,1,1,1,1,0,0,0],10,'J',6,'I')  # training data list 610, letter 'J', var2, courtesy JS, Cohort 3; serif 


# Training data with unusual variants, e.g. scripts (numbering from 701 ++)

    trainingDataListAz1 = (701,[0,0,0,1,1,1,1,0,0, 0,0,1,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,1,0, 0,1,0,0,0,0,1,1,0, 0,0,1,0,0,1,0,1,0, 0,0,0,1,1,0,0,0,1],1,'A', 1,'A') # training data list 28, letter 'A', variant 2, courtesy JBA, Cohort 3 - script lower 'a'



#    trainingDataListG1  = (7,[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,1,1,1,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,1,0],7,'G',3,'C')    
#    trainingDataListQ = (17,[0,0,1,1,1,1,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,0,1,0,1,0,0,0,0,0,1,0,0,0,1,1,1,1,1,0,1],17,'Q') # training data list 17, letter 'Q', courtesy RG    
#    trainingDataListE1 = (27,[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],5,'E') # training data VARIANT for 'E,' courtesy BMcD
#    trainingDataListM1 = (28,[1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,1,1,0,0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1],13,'M') # training data list 28, variant for 'M', courtesy TD
                      
#    trainingDataListE =(5,[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],5,'E') # training data VARIANT for 'E,' courtesy BMcD   
#    trainingDataListE =(5,[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,1,1,1,1,1,1,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],5,'E') # training data VARIANT for 'E,' courtesy BMcD - serif

#trainingDataListH = (8,[1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0 Jathin C R
#trainingDataListH = (8,[1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0

# Kevin Wong
# trainingDataListw = (23, [1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,0,0,1,0,1,1,1,0,0,0,0,0,1,1],23,'W')
# trainingDataListw = (23, [1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,1,0,1,0,0,1,1,0,1,0,0,0,1,0,1,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1],23,'W')
# trainingDataListw = (23, [1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,1,0,1,0,1,0,1,0,1,0,0,0,1,0,0,0,1,0,0],23,'W')

# Serif M Chang
# trainingDataListw = (13,[1,1,0,0,0,0,0,1,1,0,1,0,0,0,0,0,1,0,0,1,1,0,0,0,1,1,0,0,1,0,1,0,1,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,1,1,0,0,0,1,1,1],13,'M')

# Sameera
# trainingDataListF = (6,[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],6,'F')

# Troy D: M, O, D
# trainingDataListM = (13,[1,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,1,0,1,0,0,0,1,0,1,1,0,0,1,0,1,0,0,1,1,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1],13,'M') # training data list 13 selected for the letter 'M'
# trainingDataListO = (15,[0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,0],15,'O') # training data list 15 selected for the letter 'O'
# trainingDataListD = (4,[1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0],4,'D') # training data list 4 selected for the letter 'D'

# Ishmael A. 
# trainingDataListX = (18,[1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1],18,'R')
# trainingDataListR2= (18,[1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1],18,'R')

# Orion
# trainingDataListO = (15,[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,],15,'O')    
 
# Michael T. 
# trainingDataListP = (16,[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],16,'P')       
# trainingDataListP = (16,[0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],16,'P')
# trainingDataListP = (16,[0,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,1,0,0,1,0,0,0,1,0,1,0,0,1,0,0,0,1,0,1,0,0,1,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],16,'P')    

# Robert G.
# trainingDataListG = (7,[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],7,'G')

# trainingDataListG = (7,[1,1,1,1,1,1,1,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,1,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1,1,0],7,'G')

# trainingDataListG = (7,[0,0,1,1,1,1,0,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,1,1,1,0,0,0],7,'G')

# trainingDataListG = (7,[1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,0,0,0,1,1,0,1,1,0,0,0,0,0,0,0,1,1,0,0,1,1,1,1,0,1,1,0,0,1,1,1,1,0,1,1,0,0,0,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,1,1,1,1,1,1,0],7,'G')

# trainingDataListG = (7,[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,1,1,1,1,1,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],7,'G')
    
# Richard G.
# TrainingDataListS=(19,[1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1],19,'S')

# Justice D. - U
# trainingDataListU = (21,[1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0, 1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,1,1,0,1,1,0,0,0,1,1, 0,0,0,1,1,1,1,1,0,0],21,'U')
 
# TrainingDataListS=(19,[0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0],19,'S')
# TrainingDataListS=(19,[0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0],19,'S')
# TrainingDataListS = (19,[0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,0],19,'S')

# TrainingDataListS = (19,[0,1,1,1,1,1,1,1,0,1,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0,0,1,1,0,1,1,1,1,1,1,1,0],19,'S')

# Erik O. - K

# TrainingDataListS = (11,[1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,1,1,0,1,0,0,0,1,1,0,0,0,1,0,1,1,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,1,0,0,0,0,0,1,1,0,1,0,0,0,0,0,0,0,1],11,'K')            
# TrainingDataListS = (11,[1,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,0,1,0,0,0,0,1,1,0,0,1,0,0,1,1,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,1,1,0,0,0,0,1,0,0,0,0,1,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,1],11,'K')
# TrainingDataListS = (11,[1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0],11,'K')
# Erik O - variants with offset main vertical
# TrainingDataListS = (11,[1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0],11,'K')                
# TrainingDataListS = (11,[0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0],11,'K')                                
# TrainingDataListS = (11,[0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1],11,'K')
# TrainingDataListS = (11,[0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0],11,'K')
# TrainingDataListS = (11,[0,0,1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,1,0,0,0,0,0,1,1,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1],11,'K')                                                               

                                                                                                

    
    if dataSet == 1: trainingDataList = trainingDataListA0
    if dataSet == 2: trainingDataList = trainingDataListB0 
    if dataSet == 3: trainingDataList = trainingDataListC0
    if dataSet == 4: trainingDataList = trainingDataListD0     
    if dataSet == 5: trainingDataList = trainingDataListE0
    if dataSet == 6: trainingDataList = trainingDataListF0 
    if dataSet == 7: trainingDataList = trainingDataListG0 
    if dataSet == 8: trainingDataList = trainingDataListH0
    if dataSet == 9: trainingDataList = trainingDataListI0
    if dataSet == 10: trainingDataList = trainingDataListJ0    
    if dataSet == 11: trainingDataList = trainingDataListK0            
    if dataSet == 12: trainingDataList = trainingDataListL0
    if dataSet == 13: trainingDataList = trainingDataListM0
    if dataSet == 14: trainingDataList = trainingDataListN0 
    if dataSet == 15: trainingDataList = trainingDataListO0 
    if dataSet == 16: trainingDataList = trainingDataListP0   
    if dataSet == 17: trainingDataList = trainingDataListQ0
    if dataSet == 18: trainingDataList = trainingDataListR0 
    if dataSet == 19: trainingDataList = trainingDataListS0        
    if dataSet == 20: trainingDataList = trainingDataListT0
    if dataSet == 21: trainingDataList = trainingDataListU0 
    if dataSet == 22: trainingDataList = trainingDataListV0   
    if dataSet == 23: trainingDataList = trainingDataListW0
    if dataSet == 24: trainingDataList = trainingDataListX0 
    if dataSet == 25: trainingDataList = trainingDataListY0        
    if dataSet == 26: trainingDataList = trainingDataListZ0
        


# Rounded training data
    if dataSet == 101: trainingDataList = trainingDataListAr1 
    if dataSet == 116: trainingDataList = trainingDataListPr1           
    if dataSet == 125: trainingDataList = trainingDataListYr1
            
# Serif training data
    if dataSet == 208: trainingDataList = trainingDataListHs1      
    if dataSet == 238: trainingDataList = trainingDataListHs1

    if dataSet == 210: trainingDataList = trainingDataListJs1  
            
    if dataSet == 223: trainingDataList = trainingDataListWs1      
    if dataSet == 225: trainingDataList = trainingDataListYs1
       

# Translated (up/down) or shifted (e.g., w/ letter 'J') or squeezed data
    if dataSet == 308: trainingDataList = trainingDataListHt1
    
    if dataSet == 311: trainingDataList = trainingDataListKt1 
    if dataSet == 341: trainingDataList = trainingDataListKt2     
    if dataSet == 371: trainingDataList = trainingDataListKt3 
    if dataSet == 372: trainingDataList = trainingDataListKt4  
    if dataSet == 373: trainingDataList = trainingDataListKt5 
    if dataSet == 374: trainingDataList = trainingDataListKt6      

    if dataSet == 312: trainingDataList = trainingDataListLt1 
    if dataSet == 342: trainingDataList = trainingDataListLt2 
   
            
    if dataSet == 316: trainingDataList = trainingDataListPt1      
    if dataSet == 325: trainingDataList = trainingDataListYt1
                                                     
# Proportions changed (up/down on vertices, e.g. w/ letters 'Y' or 'W')  (numbering from 401 ++)

    if dataSet == 406: trainingDataList = trainingDataListFx1      
    if dataSet == 435: trainingDataList = trainingDataListFx2 

# Affine transformation

    if dataSet == 501: trainingDataList = trainingDataListAa1      
    if dataSet == 531: trainingDataList = trainingDataListAa2 

# Double-thickness strokes (numbering from 601 ++)

    if dataSet == 610: trainingDataList = trainingDataListJd1  


# Unusual variants, e.g. scripts (numbering from 701 ++)

    if dataSet == 701: trainingDataList = trainingDataListAz1 
                                                                                                                                                                                                                            
                                                                                                                
    return (trainingDataList) 


####################################################################################################
####################################################################################################
#
# Procedure to print out a letter, given the number of the letter code
#
####################################################################################################
####################################################################################################

def printLetter (trainingDataList):    
            
    pixelArray = trainingDataList[1]
    print ' '
    gridWidth = 9
    gridHeight = 9
    iterAcrossRow = 0
    iterOverAllRows = 0
    while iterOverAllRows <gridHeight:
        while iterAcrossRow < gridWidth:
            arrayElement = pixelArray [iterAcrossRow+iterOverAllRows*gridWidth]
            if arrayElement <0.9: printElement = ' '
            else: printElement = 'X'
            print printElement, 
            iterAcrossRow = iterAcrossRow+1
        print ' '
        iterOverAllRows = iterOverAllRows + 1
        iterAcrossRow = 0 #re-initialize so the row-print can begin again
    print 'The data set is for the letter', trainingDataList[3], ', which is alphabet number ', trainingDataList[2]
    if trainingDataList[0] > 30: 
        print 'This is a variant pattern for letter ', trainingDataList[3]

    
    return     

####################################################################################################
####################################################################################################
#
# Function to obtain the user's choice of desired action
#
####################################################################################################
####################################################################################################

def obtainUserChoice():

    userChoice = 1 
    print ' '
    print 'Letter selection: '
    print ' '                     
    print 'Do you wish to view the grid pattern for a specific letter (1) or see a specific variant (0)?'
    userInput = input('Please enter 1 or 0: ')
    if userInput == 1: userChoice = userInput
    else:
        print ' '
        print 'Please give the exact (numeric) key for the desired variant'
        userChoice = input('Variant key: ')
        
    return userChoice                                                                 



####################################################################################################
####################################################################################################
#
# Function to obtain the user's specific choice of letter
#
####################################################################################################
####################################################################################################

def obtainChosenLetter(letterType):
    letterNum = 1

    if letterType == 1: letterNum = obtainSpecificPrimaryLetter()
    else: letterNum == -1
        
    return letterNum                                                                 


####################################################################################################
####################################################################################################
#
# Function to identify the specific variant type
#
####################################################################################################
####################################################################################################

def findVariantType(userChoice):
    variantType = 1

    if userChoice > 700: variantType = 7 # Unusual variant; not easily typed (e.g., script letter)
    else: 
        if userChoice > 600: variantType = 6 # Double-wide variant
        else:
            if userChoice > 500: variantType = 5 # Affine variant
            else:
                if userChoice > 400: variantType = 4 # Proportions changed variant
                else:
                    if userChoice > 300: variantType = 3 # Translated/shifted variant
                    else:
                        if userChoice > 200: variantType = 2 # Serif variant
                        else: variantType = 1 # Rounded variant
        
    return variantType   
                                                                                                                                                                                                      


####################################################################################################
####################################################################################################
#
# Function to obtain the user's specific letter choice 
#
####################################################################################################
####################################################################################################

def obtainSpecificPrimaryLetter():
    letterChoice = 'A'
    print ' '
    print ' ==>> Remember to put single quotes around your letter!'
    letterChoice = input ('Enter the capital letter that you wish to print out in grid format: ')
    if letterChoice == 'A': letterNum = 1
    if letterChoice == 'B': letterNum = 2
    if letterChoice == 'C': letterNum = 3   
    if letterChoice == 'D': letterNum = 4
    if letterChoice == 'E': letterNum = 5
    if letterChoice == 'F': letterNum = 6   
    if letterChoice == 'G': letterNum = 7
    if letterChoice == 'H': letterNum = 8
    if letterChoice == 'I': letterNum = 9   
    if letterChoice == 'J': letterNum = 10
    if letterChoice == 'K': letterNum = 11
    if letterChoice == 'L': letterNum = 12    
    if letterChoice == 'M': letterNum = 13
    if letterChoice == 'N': letterNum = 14
    if letterChoice == 'O': letterNum = 15   
    if letterChoice == 'P': letterNum = 16
    if letterChoice == 'Q': letterNum = 17
    if letterChoice == 'R': letterNum = 18   
    if letterChoice == 'S': letterNum = 19
    if letterChoice == 'T': letterNum = 20
    if letterChoice == 'U': letterNum = 21   
    if letterChoice == 'V': letterNum = 22
    if letterChoice == 'W': letterNum = 23
    if letterChoice == 'X': letterNum = 24              
    if letterChoice == 'Y': letterNum = 25
    if letterChoice == 'Z': letterNum = 26

       
    print ' The letter chosen is :', letterChoice, 'which is number ', letterNum
    return letterNum
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
####################################################################################################
#**************************************************************************************************#
####################################################################################################


def main():

####################################################################################################
# Obtain a training data set
####################################################################################################                

# This calls the function 'obtainRandomAlphabetTrainingValues (),' which returns one of the several
#   training data sets. 
#
# Tutorial notes:
# All procedures and functions need an argument list. 
# This function has a list, but it is an empty list; obtainRandomAlphabetTrainingValues ().

# This defines the variable trainingDataList, which is a list. It is initially an empty list. 
# Its purpose is to store four different elements that are important in each training data set.

    trainingDataList = list() # empty list
    userChoice = obtainUserChoice()
    print '  UserChoice = ', userChoice
    if userChoice > 30: dataSet = userChoice

    if userChoice == 1: 
        letterNum = obtainSpecificPrimaryLetter()
        print '  You have requested a specific letter'
        print '    The letter corresponds to number ', letterNum
        dataSet = letterNum        

# We return the list from the function, with values placed inside the list.    

    trainingDataList = obtainAlphabetTrainingValues (dataSet)
    print ' '
    print 'The training data set is ', trainingDataList[0]

# The next step will be to print the list

    printLetter(trainingDataList)
    if userChoice >30:
        dataSet = userChoice
        variantType = findVariantType(userChoice)
        if variantType == 1: print '  This is variant type: Rounded'
        if variantType == 2: print '  This is variant type: Serif'
        if variantType == 3: print '  This is variant type: Translated/shifted'
        if variantType == 4: print '  This is variant type: Proportions changed'
        if variantType == 5: print '  This is variant type: Affine'
        if variantType == 6: print '  This is variant type: Double-thickness'
        if variantType == 7: print '  This is variant type: Non-specific (e.g., script - other)'    

# Write the list to a file

    test = open('Canopy/datafiles/testfile1', 'w')

    for item in trainingDataList:
        test.write("%s\n" % item)  
 
    test.close()
    
    
 


                     
####################################################################################################
# Conclude specification of the MAIN procedure
####################################################################################################                
    
if __name__ == "__main__": main()

####################################################################################################
# End program
#################################################################################################### 
# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sns
sns.set(color_codes=True)
import pandas as pd
import matplotlib.pyplot as plt
import string
import hdbscan


np.set_printoptions(precision=3)
#Input activations corresponding to X, M, N, H, A respectively

x = np.array([
[0,0,0,0,1,0,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,1,0,0,0,1,0,0, 0,1,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],
[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,0, 1,1,1,1,1,1,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,0],
[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,0, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1,1,0],
[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],
[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1],
[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1],
[0,0,1,1,1,1,1,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,1,1,1,1,1,0,0],
[0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,1,1,0,0,0],
[1,0,0,0,0,0,1,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,1,1,0,0,0,0,0,0, 1,0,0,1,0,0,0,0,0, 1,0,0,0,1,0,0,0,0, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,0,1,0,0],
[1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1],
[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,0,1,0,0,0,1,0,1, 1,0,1,0,0,0,1,0,1, 1,0,0,1,0,1,0,0,1, 1,0,0,1,0,1,0,0,1, 1,1,0,0,1,0,0,0,1, 1,0,0,0,1,0,0,0,1],      
[1,0,0,0,0,0,0,0,1, 1,1,0,0,0,0,0,0,1, 1,0,1,0,0,0,0,0,1, 1,0,0,1,0,0,0,0,1, 1,0,0,0,1,0,0,0,1, 1,0,0,0,0,1,0,0,1, 1,0,0,0,0,0,1,0,1, 1,0,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1],
[0,1,1,1,1,1,1,1,0, 1,1,0,0,0,0,0,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 0,1,1,1,1,1,1,1,0],
[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0],
[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,1,0,0,1, 1,0,0,0,0,0,1,0,1, 1,0,0,0,0,0,0,1,1, 1,1,1,1,1,1,1,1,1],
[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1, 1,0,0,0,0,1,0,0,0, 1,0,0,0,0,0,1,0,0, 1,0,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1],
[1,1,1,1,1,1,1,1,1, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,1, 0,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1,1],
[0,1,1,1,1,1,1,1,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,0,1,0,0,0,0],
[1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 1,1,0,0,0,0,0,1,1, 0,1,1,0,0,0,1,1,0, 0,0,1,1,1,1,1,0,0],
[1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,0,0,0,0,1, 1,0,0,0,1,0,0,0,1, 1,0,0,1,0,1,0,0,1, 1,0,1,0,0,0,1,0,1, 0,1,0,0,0,0,0,1,0],
[1,0,0,0,0,0,0,0,1, 0,1,0,0,0,0,0,1,0, 0,0,1,0,0,0,1,0,0, 0,0,0,1,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,1,0,1,0,0,0, 0,0,1,0,0,0,1,0,0, 0,1,0,0,0,0,0,1,0, 1,0,0,0,0,0,0,0,1],
[1,1,1,1,1,1,1,1,1, 0,0,0,0,0,0,0,1,0, 0,0,0,0,0,0,1,0,0, 0,0,0,0,0,1,0,0,0, 0,0,0,0,1,0,0,0,0, 0,0,0,1,0,0,0,0,0, 0,0,1,0,0,0,0,0,0, 0,1,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1]
])


y = x

#Calculates the total error or the derivative of the total error
def errorCalc(yTrue, yHat, deriv = False):
    """
    :param yTrue: actual/target values
    :param yHat: predicted values
    :param deriv: False (default) if you want the error; True if you want the derivative of the error
    """
    if(deriv==True):
        return -(yTrue-yHat)
    else:
        return np.sum(0.5 * (yTrue-yHat)**2.0)

#The 'transfer' function
def transfer(x,alpha = 1.0, deriv=False, method = "sigmoid"):
    if method == "sigmoid":
        if(deriv==True):
            return alpha*x*(1.0-x)
        else:
            return 1.0/(1.0+np.exp(-alpha*x))
    
    elif method == "tanh":
        if(deriv==True):
            return alpha*(1.0-np.tanh(alpha*x)**2)
        else:
            return np.tanh(alpha * x)
    elif method == "ReLU":
        if(deriv==True):
            return alpha * np.exp(alpha * x)/(1 + np.exp(alpha*x))
        else:
            return np.log(1+np.exp(alpha*x))

def updatedConfig(recentChange, direction, currentConfig):
    """
    :param recentChange: change in nodes vs previous attempt
    :param direction: 1 in error went down, -1 if error went up
    :param currentConfig: the current, full network configuration as a list
    """    
    probs = {3: [0.9,0.1], 0: [0.5,0.5], -3: [0.1,0.9]} #probability of increasing or decreasing hidden node cnt
    newConfig = currentConfig
    loc = np.random.choice(np.arange(1,len(currentConfig)-1)) #Which layer to change
    delta = np.random.choice([3,-3], p=probs[recentChange * direction]) #Increase or decrease nodes

    newConfig[loc] += delta        
    return newConfig

#Configuration / architecture
config = [81,20,2,20,81] #Number of units at each layer (including input and output layer)
layerCnt = len(config) #Number of layers

#Learning Rate
eta = 0.7
priorEta = eta
etaChange = eta-priorEta

#Transfer function parameter
alphaSigmoid = 0.5
priorAlphaSigmoid = alphaSigmoid
alphaChangeSigmoid = alphaSigmoid - priorAlphaSigmoid

alpha = 0.4
priorAlpha = alpha
alphaChange = alpha - priorAlpha

changeDirection = 0.0

nodeCnt = sum(config[1:layerCnt-1])
priorNodeCnt = nodeCnt
nodeCntChange = nodeCnt - priorNodeCnt


maxError = 30 #Maximum allowable error
errorTotal = maxError + 5
errorPrior = errorTotal

#Model attributes
weights = {} #Store the weights in every layer
errorDerivs = {}
deltas = {}

#Store model states over time
errorHist = [] #History of the error for plotting
errorFinalHist = pd.DataFrame({'attempt': 0, 'nodes': nodeCnt, 'alpha': alpha,'alphaSigmoid': alphaSigmoid,'eta': eta,'error': errorTotal}, index = [0]) #History of final errors at the end of each attempt
weightFinalHist = [] #History of final weights at the end of each attempt
weightsByLevels = []

#Keep trying new random weights until you converge
attempts = 0
while errorTotal > maxError and attempts <= 25:
    #Reseet error/weight hist
    errorHist = []
    weightHist = []
    
    if attempts == 0: #Do not perturb hyperparameters
        changeDirection = 0.0        
    else:
        #Move in the direction of error minimization
        if errorTotal < errorPrior:
            changeDirection = 1.0
        else:
            changeDirection = -1.0

        nodeCntChange = nodeCnt-priorNodeCnt
        alphaChange = alpha - priorAlpha
        alphaChangeSigmoid = alphaSigmoid - priorAlphaSigmoid
        etaChange = eta - priorEta
        errorPrior = errorTotal
        
        priorAlpha = alpha
        priorAlphaSigmoid = alphaSigmoid
        priorEta = eta
        priorNodeCnt = nodeCnt
        
        #Update weights in the error minimization direction + random perterbation
        alphaSigmoid += np.random.normal(loc = alphaChangeSigmoid*changeDirection, scale = alphaSigmoid*0.1, size = 1)
        alpha += np.random.normal(loc = alphaChange*changeDirection, scale = alpha*0.1, size = 1)
        eta += np.random.normal(loc = etaChange*changeDirection, scale = eta* 0.1,size = 1)                        
        
#        config = updatedConfig(nodeCntChange,changeDirection, config)
        nodeCnt = sum(config[1:layerCnt-1])

        #Hyperparameters must be positive
        alphaSigmoid = np.take(np.max([alphaSigmoid,0.001]),0)
        alpha = np.take(np.max([alpha,0.001]),0)
        eta = np.take(np.max([eta,0.001]),0)

    #Set random initial weights (from -1 to 1, centered at 0)
#    np.random.seed(1)
    for layer in np.arange(1,layerCnt,1):
        currentLayerSize = config[layer]    
        priorLayerSize = config[layer-1]
        weights[layer] = (np.random.random((priorLayerSize,currentLayerSize))-0.5)
    
    #Iterations
    i = 0
    while i < 5000 and np.abs(np.sum(errorTotal)) > maxError:
        fwdPass = x
        outputs = {0: fwdPass} #This dictionary will hold output of every layer (including input layer)
    
        #Forward pass
        for level in np.arange(1,layerCnt,1):
            fwdPass = np.dot(fwdPass,weights[level]) #Apply weights
        
            if level == layerCnt-1: #i.e. the final output layer
                fwdPass = transfer(fwdPass, alpha = alphaSigmoid, method = "sigmoid") #Apply transfer                
            else:
                fwdPass = transfer(fwdPass, alpha = alpha, method = "sigmoid") #Apply transfer

            outputs[level] = fwdPass #Store level ouput; to be used in deriv of backprop algorithm    
    
        #Backward pass
        errorTotal = errorCalc(y, fwdPass, False)
        errorHist.append(errorTotal) 
        
        #Backpass
        for layer in reversed(np.arange(1,layerCnt)):
            if layer == layerCnt-1: #i.e. the final output layer
                errorDerivs[layer] = errorCalc(y, fwdPass, True) #Derivative of error wrt final output
                deltas[layer] = errorDerivs[layer] * transfer(outputs[layer], alpha = alphaSigmoid, deriv = True, method = "sigmoid")            
            else:
                errorDerivs[layer] = deltas[layer+1].dot(weights[layer+1].T)
                deltas[layer] = errorDerivs[layer] * transfer(outputs[layer],alpha = alpha, deriv = True, method = "sigmoid")  


        #Update weights
        for layer in reversed(np.arange(1,layerCnt)):
            weights[layer] -= eta*outputs[layer-1].T.dot(deltas[layer])    
    
        #Only store iteration-level level at a certain frequency
        if i % 10 == 0:
            errorHist.append(np.abs(errorTotal))
                    
        i += 1

    #Store the error at the end of each attempt        
    errorFinalHist = errorFinalHist.append({'attempt': attempts, 'nodes': nodeCnt,'alpha': alpha,'alphaSigmoid': alphaSigmoid,'eta': eta,'error': errorTotal}, ignore_index = True)

    #Report back progress towards attempted convergence
    if attempts % 1 == 0:
        print("Attempt: ", attempts,"i: ", i, "Error: ", np.round(errorTotal,3))
        print('nodeCnt: ', nodeCnt, 'alpha: ', np.round(alpha,3), 'alphaSigmoid: ', np.round(alphaSigmoid,3), 'eta: ', np.round(eta, 3))
        print('')
    attempts += 1

print('')
print("Final Result:")
print("Attempt: ", attempts-1,"i: ", i, "Error: ", np.round(errorTotal,3))
print('nodeCnt: ', nodeCnt, 'alpha: ', np.round(alpha,3), 'alphaSigmoid: ', np.round(alphaSigmoid,3), 'eta: ', np.round(eta, 3))

#Graphs
#Weight/error evoluation over time
#errorHist = pd.DataFrame(errorHist)
#errorHist.plot(title = "SSE Progression in Final Model", legend = False)
#weightHist.plot(title = "Progression of Weights in Final Model", legend = False, alpha = 0.8)


#sns.kdeplot(weights[1].flatten(), alpha = 0.75, label = 'Layer 1')
#sns.kdeplot(weights[2].flatten(), alpha = 0.75, label = 'Layer 2')
#sns.kdeplot(weights[3].flatten(), alpha = 0.75, label = 'Layer 3')
#sns.kdeplot(weights[4].flatten(), alpha = 0.75, label = 'Layer 4')

labels = list(string.ascii_uppercase)
labels.pop(24)
labels.pop(21)

encoded = pd.DataFrame(outputs[2], columns = ['l1','l2'], index = labels)
#l = encoded['l3']
#l = (l-np.min(l))/(np.max(l)-np.min(l))*700+50
#encoded.plot(kind = 'scatter', x = 'l2', y='l3')

clusterer = hdbscan.HDBSCAN(min_cluster_size = 2)
clusterer.fit(encoded)
colors = clusterer.labels_

sns.color_palette('deep', np.unique(colors).max() + 1)

fig, ax = plt.subplots()
encoded.plot(kind = 'scatter', x = 'l1', y='l2', c = colors,s = 100, legend = True, ax = ax, title = 'Encoded Letters (Size/Color = l3)')
np.random.seed(2)
for k, v in encoded.iterrows():
#    print(k,v['l1'])    
    ax.annotate(k, xy = (v['l1']+np.random.uniform(-0.03,0.03),v['l2']+np.random.uniform(-0.03,0.03)), size = 15, color = 'blue')

#sizes = 150*(errorFinalHist['error']-errorFinalHist['error'].min())/(errorFinalHist['error'].max()-errorFinalHist['error'].min())
#sns.pairplot(errorFinalHist, hue = 'attempt', plot_kws={"s": sizes})
#errorFinalHist.plot(x='attempt', y = ['nodes','error'])
#errorFinalHist.plot(x='attempt', y = ['alpha','alphaSigmoid','eta'])

#errorFinalHist.plot(title = "Distribution of Final Error Across Attempts", legend = False)
import numpy as np

#Create function for manual multiplication
def MultiplyVector(vec1, vec2):
    elementProduct = vec1 * vec2
    return (elementProduct)

# Set vectors arrays values
v1_0 = 2
v1_1 = 2
v1_2 = 2
v1_3 = 2
v1_4 = 2
v1 = [v1_0, v1_1, v1_2, v1_3, v1_4]
Array1 = np.array(v1)

v2_0 = 4
v2_1 = 4
v2_2 = 4
v2_3 = 4
v2_4 = 4
v2 = (v2_0, v2_1, v2_2, v2_3, v2_4)
Array2 = np.array(v2)

print 'Vector arrays:'
print ' Array 1 = ', Array1
print ' Array 2 = ', Array2

########################################################################
# Method 1: Use loop to manually multiply the value in array
########################################################################
i = 0
manualProduct = 0
manualLoopProduct = 0
arrayLen = len(v1)

while i < arrayLen:
    manualProduct = MultiplyVector(v1[i], v2[i])
    manualLoopProduct = manualLoopProduct + manualProduct
    i = i + 1

print 'Method 1: Use loop to manually multiply the value in array'
print ' Dot Product = ', manualLoopProduct

########################################################################
# Method 2: Directly use an array dot product function
########################################################################
DotProduct = np.dot(Array1,Array2)

print '# Method 2: Directly use an array dot product function'
print ' Dot Product = ', DotProduct
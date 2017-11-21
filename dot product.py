import numpy as np


# random function to add two values together
def function_1(input_1, input_2):
    value_1 = input_1 + input_2
    return value_1


List1 = np.array([2, 2, 2, 2])  # first array
List2 = np.array([1, 1, 1, 1])  # second array
List3 = np.zeros(len(List1))  # blank array

print(List1)
print(List2)
print(List3)

i = 0  # set to 0 to begin while loop

while i < 2:  # the while loop can be changed to a desired number of loops
    i = i + 1
    List1 = np.multiply(List1, List2)
    List2 = List1 + 1
    print(List1)
    print(List2)

    for i in range(0, len(List3)):  # for loop to access each value in the array
        List3[i] = function_1(List1[i], List2[i])  # redefining the values in List 3 based on function_1

DotProd = np.dot(List1, List2)  # dot product calculator
print(DotProd)

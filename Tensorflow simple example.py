import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

training_data = np.array([[1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],
                          [1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                          [1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1],
                          [1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1],
                          [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1]])
# fullSet.append([1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 5, 'I'])
# fullSet.append([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1, 6, 'L'])
# fullSet.append([1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 7, 'B'])

# the 5 expected results in same order
target_data = np.array([[0], [1], [2], [3], [4]])

model = Sequential()
# Add 6 neurons to the hidden layer (relu activation gets to correct answer in 375 iterations, sigmoid takes 637)
model.add(Dense(6, input_dim=25, activation='sigmoid'))
# model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(training_data, target_data, nb_epoch=1000, verbose=2)

print(model.predict(training_data).round())

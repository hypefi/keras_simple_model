#scripting scripting 

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense

# load the dataset 
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# split into ipnut X and output Y variable 

X = dataset[:,0:8]
y = dataset[:,8]

print(X)
print(y)

#defining the model 

model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

#compile the keras model 
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the keras model on the dataset

model.fit(X,y, epochs=150, batch_size=10)

# evaluate the model 
_, accuracy = model.evaluate(X,y)
print('Accuracy: %.2f' % (accuracy*100))

# make class predictions with the model
predictions = model.predict_classes(X)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

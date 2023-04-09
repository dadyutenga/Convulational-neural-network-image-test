from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Define the CNN model
model = Sequential()

# Add a convolutional layer with 32 filters, each of size 3x3, and activation function 'relu'
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))

# Add a max pooling layer with pool size 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add another convolutional layer with 64 filters, each of size 3x3, and activation function 'relu'
model.add(Conv2D(64, (3, 3), activation='relu'))

# Add another max pooling layer with pool size 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the feature maps into a 1D vector
model.add(Flatten())

# Add a fully connected layer with 128 units and activation function 'relu'
model.add(Dense(128, activation='relu'))

# Add an output layer with the number of classes in your dataset and activation function 'softmax'
model.add(Dense(num_classes, activation='softmax'))

# Compile the model with appropriate loss function, optimizer, and evaluation metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model with your training data
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

# Evaluate the model with your testing data
score = model.evaluate(X_test, y_test, batch_size=32)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

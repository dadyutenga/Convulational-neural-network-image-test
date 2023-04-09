from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.initializers import RandomUniform

# Create a CNN model
model = Sequential()

# Add a convolutional layer with 64 filters, each of size 3x3, and activation function 'relu'
# Initialize the weights with small positive numbers using RandomUniform initializer
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3), 
                 kernel_initializer=RandomUniform(minval=0.01, maxval=0.05)))
# Add another convolutional layer with 64 filters, each of size 3x3, and activation function 'relu'
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', 
                 kernel_initializer=RandomUniform(minval=0.01, maxval=0.05)))
# Add a max pooling layer with pool size 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add another convolutional layer with 128 filters, each of size 3x3, and activation function 'relu'
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', 
                 kernel_initializer=RandomUniform(minval=0.01, maxval=0.05)))
# Add another convolutional layer with 128 filters, each of size 3x3, and activation function 'relu'
model.add(Conv2D(128, (3, 3), activation='relu', padding='same', 
                 kernel_initializer=RandomUniform(minval=0.01, maxval=0.05)))
# Add a max pooling layer with pool size 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the feature maps into a 1D vector
model.add(Flatten())

# Add a fully connected layer with 256 units and activation function 'relu'
model.add(Dense(256, activation='relu', 
                kernel_initializer=RandomUniform(minval=0.01, maxval=0.05)))
# Add an output layer with the number of classes in your dataset and activation function 'softmax'
model.add(Dense(num_classes, activation='softmax'))

# Compile the model with appropriate loss function, optimizer, and evaluation metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

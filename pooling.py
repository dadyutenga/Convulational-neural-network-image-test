from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D

# Create a CNN model
model = Sequential()

# Add a convolutional layer with 64 filters, each of size 3x3, activation function 'relu',
# and padding='same' to maintain the input size
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3), strides=1))

# Add a max pooling layer with pool size 2x2 to reduce the spatial dimensions by half
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add more convolutional and pooling layers as needed

# Flatten the feature maps into a 1D vector
model.add(Flatten())

# Add fully connected layers and output layer as needed

# Compile the model with appropriate loss function, optimizer, and evaluation metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

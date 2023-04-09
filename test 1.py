from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Create a CNN model
model = Sequential()

# Add a convolutional layer with 64 filters, each of size 3x3, and activation function 'relu'
model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(64, 64, 3)))
# Add batch normalization to normalize the input
model.add(BatchNormalization())
# Add another convolutional layer with 64 filters, each of size 3x3, and activation function 'relu'
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
# Add batch normalization
model.add(BatchNormalization())
# Add a max pooling layer with pool size 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Add dropout to prevent overfitting
model.add(Dropout(0.25))

# Add another convolutional layer with 128 filters, each of size 3x3, and activation function 'relu'
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# Add batch normalization
model.add(BatchNormalization())
# Add another convolutional layer with 128 filters, each of size 3x3, and activation function 'relu'
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
# Add batch normalization
model.add(BatchNormalization())
# Add a max pooling layer with pool size 2x2
model.add(MaxPooling2D(pool_size=(2, 2)))
# Add dropout
model.add(Dropout(0.5))

# Flatten the feature maps into a 1D vector
model.add(Flatten())

# Add a fully connected layer with 256 units and activation function 'relu'
model.add(Dense(256, activation='relu'))
# Add batch normalization
model.add(BatchNormalization())
# Add dropout
model.add(Dropout(0.5))

# Add an output layer with the number of classes in your dataset and activation function 'softmax'
model.add(Dense(num_classes, activation='softmax'))

# Compile the model with appropriate loss function, optimizer, and evaluation metric
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

# Apply data augmentation to generate new training images on-the-fly
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical')

# Train the model with augmented training data
model.fit_generator(
    train_generator,
    steps_per_epoch=num_train_samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=num_val_samples // batch_size)

# Evaluate the model with your testing data
score = model.evaluate_generator(test_generator, steps=num_test_samples // batch_size)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

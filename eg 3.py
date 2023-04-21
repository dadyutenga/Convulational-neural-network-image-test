from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load your dataset and preprocess it as needed

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Create an SVM model with a linear kernel
model = svm.SVC(kernel='linear')

# Train the model on the training set
model.fit(X_train, y_train)

# Predict the labels of the testing set using the trained model
y_pred = model.predict(X_test)

# Calculate the accuracy of the model on the testing set
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))

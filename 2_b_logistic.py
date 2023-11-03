import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# For simplicity, let's only consider binary classification (Setosa vs not-Setosa)
y = (y == 0).astype(int)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (mean=0 and variance=1)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

D = X_train.shape[1]  # Number of dimensions

def gradient(matrix, w):
    Y = matrix[:, 0]  # point labels
    X = matrix[:, 1:]  # point coordinates
    return ((1.0 / (1.0 + np.exp(-Y * X.dot(w))) - 1.0) * Y * X.T).sum(1)

def add(x, y):
    x += y
    return x

def logistic_regression(X_train, y_train, iterations):
    train_matrix = np.column_stack((y_train, X_train))
    w = 2 * np.random.ranf(size=D) - 1  # Initialize w to a random value
    
    for i in range(iterations):
        w -= gradient(train_matrix, w)
    
    return w

def calculate_accuracy(X, y, w):
    predictions = 1 / (1 + np.exp(-X.dot(w))) >= 0.5
    accuracy = (predictions == y).mean()
    return accuracy

iterations = 100
w = logistic_regression(X_train, y_train, iterations)

train_accuracy = calculate_accuracy(X_train, y_train, w)
test_accuracy = calculate_accuracy(X_test, y_test, w)

print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

def plot_decision_boundary(X, y, w):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Here we take into account all dimensions
    Z = np.c_[np.ones(len(xx.ravel())), xx.ravel(), yy.ravel(), np.zeros(len(xx.ravel()))]  # add zeros for the other dimension
    Z = 1 / (1 + np.exp(-Z.dot(w)))
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 1], X[:, 2], c=y, edgecolors='k', marker='o', s=50, linewidth=1)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.savefig('decision_boundary.png')
    plt.show()

# Merge X_train and y_train for the plot
train_matrix = np.column_stack((y_train, X_train))
plot_decision_boundary(train_matrix, y_train, w)

# # For downloading the saved image
# import shutil
# shutil.move('decision_boundary.png', '/path/to/download/folder/decision_boundary.png')

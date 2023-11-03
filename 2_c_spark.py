from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler as SklearnScaler

# Initialize Spark Session
spark = SparkSession.builder.appName('LogisticRegressionComparison').getOrCreate()

# Load the dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
y = (y == 0).astype(int)  # For binary classification

# Convert the data to a Spark DataFrame
data = [(float(y[i]), Vectors.dense(X[i].tolist())) for i in range(len(y))]
df = spark.createDataFrame(data, ["label", "features"])

# Split the data
(train_df, test_df) = df.randomSplit([0.8, 0.2], seed=42)

# Standardize the features
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)
scalerModel = scaler.fit(train_df)
train_df = scalerModel.transform(train_df)
test_df = scalerModel.transform(test_df)

# Train Logistic Regression Model using Spark ML
lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='label', maxIter=100)
lr_model = lr.fit(train_df)
training_summary_ml = lr_model.summary

# Evaluate the model
test_results_ml = lr_model.evaluate(test_df)
train_accuracy_ml = training_summary_ml.accuracy
test_accuracy_ml = test_results_ml.accuracy

print(f'Training Accuracy (Spark ML): {train_accuracy_ml * 100:.2f}%')
print(f'Test Accuracy (Spark ML): {test_accuracy_ml * 100:.2f}%')

# Naive Logistic Regression Implementation
class NaiveLogisticRegression:
    def __init__(self, D):
        self.w = 2 * np.random.ranf(size=D) - 1  # Initialize w to a random value

    def gradient(self, matrix):
        Y = matrix[:, 0]  # point labels
        X = matrix[:, 1:]  # point coordinates
        return ((1.0 / (1.0 + np.exp(-Y * X.dot(self.w))) - 1.0) * Y * X.T).sum(1)

    def fit(self, X, y, iterations):
        matrix = np.column_stack((y, X))
        for i in range(iterations):
            self.w -= self.gradient(matrix)

    def predict(self, X):
        return 1 / (1 + np.exp(-X.dot(self.w))) >= 0.5

    def accuracy(self, X, y):
        predictions = self.predict(X)
        return (predictions == y).mean()

# Split the data for naive implementation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features for naive implementation
scaler_naive = SklearnScaler()
X_train = scaler_naive.fit_transform(X_train)
X_test = scaler_naive.transform(X_test)

# Train and evaluate the naive logistic regression model
naive_lr = NaiveLogisticRegression(D=X_train.shape[1])
naive_lr.fit(X_train, y_train, iterations=100)
train_accuracy_naive = naive_lr.accuracy(X_train, y_train)
test_accuracy_naive = naive_lr.accuracy(X_test, y_test)

print(f'Training Accuracy (Naive): {train_accuracy_naive * 100:.2f}%')
print(f'Test Accuracy (Naive): {test_accuracy_naive * 100:.2f}%')

# Comparison
print(f'Difference in Training Accuracy: {abs(train_accuracy_ml - train_accuracy_naive) * 100:.2f}%')
print(f'Difference in Test Accuracy: {abs(test_accuracy_ml - test_accuracy_naive) * 100:.2f}%')

# Stop Spark Session
spark.stop()

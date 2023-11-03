from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
import matplotlib.pyplot as plt
import numpy as np

# Initialize Spark Session
spark = SparkSession.builder.appName('LogisticRegressionExample').getOrCreate()

# Load the dataset
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

# For simplicity, let's only consider binary classification (Setosa vs not-Setosa)
y = (y == 0).astype(int)

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

# Train Logistic Regression Model
lr = LogisticRegression(featuresCol='scaledFeatures', labelCol='label', maxIter=100)
lr_model = lr.fit(train_df)
training_summary = lr_model.summary

# Evaluate the model
test_results = lr_model.evaluate(test_df)
train_accuracy = training_summary.accuracy
test_accuracy = test_results.accuracy

print(f'Training Accuracy: {train_accuracy * 100:.2f}%')
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Plot Decision Boundary (in the original feature space)
def plot_decision_boundary(lr_model):
    w = lr_model.coefficients.toArray()
    b = lr_model.intercept
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    Z = lr_model.predict(scalerModel.transform(spark.createDataFrame(
        [(Vectors.dense(feat),) for feat in np.c_[xx.ravel(), yy.ravel(), np.zeros(xx.ravel().shape[0]), np.zeros(xx.ravel().shape[0])].tolist()],
        ["features"]
    )).select("scaledFeatures").rdd.map(lambda row: row.scaledFeatures).collect())
    Z = np.array(Z).reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50, linewidth=1)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

plot_decision_boundary(lr_model)

# Stop Spark Session
spark.stop()

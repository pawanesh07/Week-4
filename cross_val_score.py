from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Create model
model = SVC(kernel='linear')

# Cross-validation
scores = cross_val_score(model, X, y, cv=5)

print("Cross Validation Scores:")
print(scores)

print("Average Accuracy:")
print(scores.mean())
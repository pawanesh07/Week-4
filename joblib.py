from sklearn.datasets import load_iris
from sklearn.svm import SVC
import joblib

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train model
model = SVC(kernel='linear')
model.fit(X, y)

# Save model
joblib.dump(model, "iris_model.pkl")

print("Model saved successfully!")
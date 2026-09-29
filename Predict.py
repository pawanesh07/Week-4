import joblib

# Load trained model
model = joblib.load("iris_model.pkl")

# New flower data
new_data = [[5.1, 3.5, 1.4, 0.2]]

# Make prediction
prediction = model.predict(new_data)

print("Predicted Class:")
print(prediction)

if prediction[0] == 0:
    print("Prediction: Setosa")
elif prediction[0] == 1:
    print("Prediction: Versicolor")
else:
    print("Prediction: Virginica")
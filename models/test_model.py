import pickle

# Load trained model
model = pickle.load(open('disease_model.pkl', 'rb'))

# Example symptoms (must match dataset column order)
sample = [[1, 1, 0, 0]]  

prediction = model.predict(sample)

print("Predicted Disease:", prediction[0])
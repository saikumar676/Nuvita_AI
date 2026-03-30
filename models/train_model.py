import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load dataset
data = pd.read_csv('../data/dataset.csv')

# Split features and target
X = data.drop('disease', axis=1)
y = data['disease']

# Train model
model = RandomForestClassifier()
model.fit(X, y)

# Save model
pickle.dump(model, open('disease_model.pkl', 'wb'))


print("Model trained successfully!")
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

data = pd.read_csv('../data/dataset.csv')

X = data.drop('disease', axis=1)
y = data['disease']

model = RandomForestClassifier()
model.fit(X, y)

pickle.dump(model, open('../models/disease_model.pkl', 'wb'))

print("Model trained successfully!")@app.route('/')
def home():
    return "Nuvita AI Backend Running 🚀"
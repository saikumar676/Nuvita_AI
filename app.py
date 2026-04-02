from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('models/disease_model.pkl', 'rb'))

@app.route('/')
def home():
    return "Nuvita AI Backend Running 🚀"

# Disease Prediction API
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    symptoms = data['symptoms']
    result = model.predict([symptoms])
    return jsonify({'disease': result[0]})

# HB Check API
@app.route('/hb', methods=['POST'])
def hb_check():
    data = request.json
    hb = float(data['hb'])
    gender = data['gender']

    if gender == "male":
        status = "Low" if hb < 13 else "Normal"
    else:
        status = "Low" if hb < 12 else "Normal"

    return jsonify({'hb_status': status}) 

# Chatbot API
@app.route('/chatbot', methods=['POST'])
def chatbot():
    data = request.json
    query = data['message'].lower()

    if "fever" in query:
        response = "You may have infection. Stay hydrated and rest."
    elif "headache" in query:
        response = "Take rest and drink water."
    else:
        response = "Please consult a doctor if symptoms continue."

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)

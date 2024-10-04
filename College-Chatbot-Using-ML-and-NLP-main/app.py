from flask import Flask, render_template, request
import pickle
import json
import random
import os

app = Flask(__name__)

# Load the trained model and vectorizer
model_path = os.path.join(os.path.dirname(__file__), 'model', 'chatbot_model.pkl')
with open(model_path, 'rb') as f:
    best_model = pickle.load(f)

model_path1 = os.path.join(os.path.dirname(__file__), 'model', 'vectorizer.pkl')
with open(model_path1, 'rb') as f:
    vectorizer = pickle.load(f)

# Load the intents data
model_path2 = os.path.join(os.path.dirname(__file__), 'dataset', 'intents1.json')
with open(model_path2, 'r') as f:
    intents = json.load(f)

def chatbot_response(user_input):
    input_text = vectorizer.transform([user_input])
    predicted_intent = best_model.predict(input_text)[0]

    # Initialize response with a default message in case no match is found
    response = "I'm sorry, I don't understand that."

    for intent in intents['intents']:
        if intent['tag'] == predicted_intent:
            response = random.choice(intent['responses'])
            break

    return response

@app.route('/')
def home():
    return render_template('main.html')  # Render the main page

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form['user_input']
        response = chatbot_response(user_input)
        return response
    return render_template('index.html')  # Render the chat page

if __name__ == '__main__':
    app.run(debug=True)

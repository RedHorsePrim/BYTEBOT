from flask import Flask, request, jsonify, render_template
import json
import os
import random
import torch
import datetime
import time
from nltk.stem import SnowballStemmer
from aimindmodel import NeuralNet
from nltk_utils import tokenize, bag_of_words
import re
import logging


app = Flask(__name__)

# Initialize logging
logging.basicConfig(filename='chatbot.log', level=logging.INFO)

# Check if required data files exist
intents_file = 'intents.json'
data_file = 'data.pth'

if not os.path.isfile(intents_file) or not os.path.isfile(data_file):
    print("Error: Required data files are missing.")
    exit()

stemmer = SnowballStemmer('english')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents and model data
with open(intents_file, 'r') as json_data:
    intents = json.load(json_data)

data = torch.load(data_file)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "BB"

# Initialize context dictionary
context = {"previous_user_input": ""}
user_profiles = {}  # User profiles for personalization

# Define conversation_log list
conversation_log = []

# Clear screen function
def clear_screen():
    os.system('cls')

# Function to preprocess user input
def preprocess_input(user_input):
    """
    Preprocess user input by tokenizing, stemming, and returning bag of words
    """
    sentence = tokenize(user_input)
    sentence_words = [stemmer.stem(word) for word in sentence]
    return bag_of_words(sentence_words, all_words)

# Conversation logging with context
def log_conversation(user_input, bot_response):
    conversation_log.append({"user_input": user_input, "bot_response": bot_response, "context": context})
    # Log user interactions for analytics
    logging.info(f"User Input: {user_input}, Bot Response: {bot_response}, Context: {context}")

# Personalization: Customize bot responses based on user profiles
def personalize_response(user_id, response):
    if user_id in user_profiles:
        user_preferences = user_profiles[user_id]
        # Customize the response based on user preferences
        # Example: You can check user_preferences and modify the response accordingly
    return response

# Function to get bot response
def get_bot_response(user_input, user_id):
    input_data = preprocess_input(user_input)
    input_data = torch.from_numpy(input_data).to(device).unsqueeze(0)

    # Update context with the current user input
    context["previous_user_input"] = user_input

    with torch.no_grad():
        output = model(input_data)
        _, predicted = torch.max(output, dim=1)
        tag = tags[predicted.item()]

        probs = torch.softmax(output, dim=1)
        prob = probs[0][predicted.item()]

        if prob.item() > 0.75:
            for intent in intents['intents']:
                if tag == intent["tag"]:
                    if intent["tag"] == "get_time_and_date":
                        now = datetime.datetime.now()
                        response = random.choice(intent["responses"])
                        if "{time}" in response:
                            response = response.format(time=now.strftime("%H:%M:%S"))
                        elif "{date}" in response:
                            response = response.format(date=now.strftime("%m/%d/%Y"))
                        return f"{bot_name}: {response}"

                    elif intent["tag"] == "timer":
                        try:
                            minutes = int(input("Enter number of minutes: "))
                            seconds = int(input("Enter number of seconds: "))
                            total_time = minutes * 60 + seconds
                            response = f"{bot_name}: Timer set for {minutes} minute(s) and {seconds} second(s)."

                            for remaining_time in range(total_time, 0, -1):
                                minutes_remaining = remaining_time // 60
                                seconds_remaining = remaining_time % 60
                                response += f"\n{bot_name}: {minutes_remaining} minute(s) and {seconds_remaining} second(s) remaining..."
                                time.sleep(1)

                            response += f"\n{bot_name}: Timer done!"
                            return response

                        except ValueError:
                            return f"{bot_name}: Invalid input. Please enter a valid number."

                    else:
                        return f"{bot_name}: {random.choice(intent['responses'])}"
        else:
            return random.choice([f"{bot_name}: Sorry, can't understand you", f"{bot_name}: I do not understand...", f"{bot_name}: Not sure I understand"])
            #return f"{bot_name}: I do not understand..."  

# Main conversation loop "WEB PAGE - التشغيل على الويب"
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")
    user_id = request.json.get("user_id")

    response = get_bot_response(user_input, user_id)

    #Personalization: Customize the response based on user profiles
    response = personalize_response(user_id, response)

    log_conversation(user_input, response)

    return jsonify({"bot_response": response})

if __name__ == "__main__":
    app.run(debug=True)
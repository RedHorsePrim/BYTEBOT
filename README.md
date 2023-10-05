## Project Title: BYTEBOT I
# Developer Name: Ahmed Abdu
# Developer Startup: Red Horse Prim
## Project Description: AI Chatbot with Flask

This GitHub project is an AI chatbot built using Python and Flask. The chatbot utilizes a neural network model to understand and respond to user inputs in a conversational manner. It has the ability to answer questions, provide the current time and date, and set timers.

### Features:

- **Natural Language Understanding:** The chatbot is capable of understanding and processing user inputs using natural language processing techniques.

- **Intent Recognition:** It recognizes user intents and responds accordingly. For example, it can answer general questions, provide the current time and date, or set timers.

- **Personalization:** The chatbot can be customized to provide tailored responses based on user profiles and preferences.

- **Conversation Logging:** All user interactions with the chatbot are logged for analytics and monitoring purposes.

### Technologies Used:

- **Flask:** The web framework used for building the chatbot's web interface.

- **PyTorch:** The neural network model is implemented using PyTorch for intent recognition.

- **NLTK:** The Natural Language Toolkit is used for text processing, including tokenization and stemming.

- **JSON:** Intents and responses are stored in JSON files for easy configuration.

### Usage:

1. Users can interact with the chatbot through a web interface by visiting the provided URL.

2. The chatbot processes user inputs, recognizes intents, and responds accordingly.

3. Personalization features can be implemented to customize responses based on user profiles.

4. All interactions and responses are logged for analysis and monitoring.

### Project Structure:

- `app.py`: The main Flask application that handles user requests and interacts with the chatbot.

- `intents.json`: JSON file containing predefined intents and responses used for intent recognition.

- `data.pth`: Data file containing model-related information, including input size, hidden size, output size, and model state.

- `aimindmodel.py`: Python module defining the neural network model used for intent recognition.

- `nltk_utils.py`: Python module containing utility functions for text processing.

- `templates/index.html`: HTML template for the chatbot web interface.

- `chatbot.log`: Log file that records all user interactions and bot responses.

### How to Run:

1. Ensure you have the required data files (`intents.json` and `data.pth`) in the project directory.

2. Install the necessary Python libraries and dependencies.

3. Run the Flask application using `app.run(debug=True)`.

4. Access the chatbot through the provided web interface.


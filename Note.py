#تنظيف الشاشة
import os
def clear_screen():
    os.system('cls')
# Clear the screen before starting the chatbot
clear_screen()


#

#---------------------------------------------------------------------------------------------#
# Main conversation loop
while True:
    user_input = input("You: ")
    user_id = "user123"  # For illustration purposes, you can replace with actual user identification

    if user_input.lower() == "quit":
        # Ask for user feedback and learn from interactions when ending the conversation
        provide_feedback(user_input, "Thank you for using the bot.")
        
        # Save conversation log
        with open('conversation_log.json', 'w') as log_file:
            json.dump(conversation_log, log_file, indent=4)

        # Save user profiles
        with open('user_profiles.json', 'w') as profile_file:
            json.dump(user_profiles, profile_file, indent=4)

        # Perform reinforcement learning model saving

        break

    response = get_bot_response(user_input, user_id)

    # Personalization: Customize the response based on user profiles
    response = personalize_response(user_id, response)

    print(response)
    log_conversation(user_input, response)


    # Machine Learning for Continuous Improvement: Update the RL model based on user feedback (Integration required)

# ... (Rest of your code)
#---------------------------------------------------------------------------------------------#

#---------------------------------------------------------------------------------------------#
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


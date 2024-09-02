# Load required libraries
import os
import warnings

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

warnings.filterwarnings('ignore')

# Fetching the API key from environment variables
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY environment variable not set.")

# Initialize Groq-LangChain
model_name = "llama3-8b-8192"  
groq_chat = ChatGroq(groq_api_key=groq_api_key, model_name=model_name)

# Initialize memory
conversational_memory_length = 5 
memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

# Set the prompt describing the instructions for conversation
system_prompt = "You are a helpful assistant that books appointments for a Dental Clinic. Your goal is to assist patients in scheduling their appointments smoothly and efficiently while maintaining a friendly tone. Greet the user. When a user expresses interest in booking an appointment, respond by asking for the specific date and time they would like to book. Once they provide this information, confirm the details by stating sentence like 'Booking your appointment for [date] at [time]... Done! Your appointment is confirmed.' Additionally, include the appointment details only if date and time is available in input and the format is: appointment_details = {'date': " "'[date]', 'time': '[time]'} After confirming, thank the patient and offer any additional assistance if needed. Keep your responses concise and to the point. Strictly do not add extra words."

# Initialize conversation history
conversation_history = []

# Define prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}"),
    ]
)

# Initialize conversation chain
conversation = LLMChain(
    llm=groq_chat,
    prompt=prompt,
    verbose=False,
    memory=memory,
)

# Set up for simple conversational flow

""" Here, the chat() function is created which does the following:
*   Handles the user's input and adds it to the conversation history as a human message
*   Setting the prompt template which includes system message i.e. instructions, our conversation history, user input and its response
*   Passing user's recent user input as per template and its corresponding variables. [Ex: Here, we have one input variable i.e. user message]
*   Giving this to LLM inorder to get the response
*   Updating our conversation history by adding the latest response to it
"""

def chat():
    """
    Initiates and manages a continuous conversation loop with the user.
    Handles user input, maintains conversation history, and generates responses using Groq and LangChain.
    """
    global conversation_history

    print("Assistant: Hi! How can I help you today?")

    while True:
      try:
        user_question = input("Patient: ")

        # Exit the loop if the user types "exit"
        if user_question.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! Have a great day!")
            break

        # Append recent user input to conversation history
        conversation_history.append({"role": "user", "content": user_question})

        # Format messages for the API
        formatted_messages = [
            {"role": "system", "content": system_prompt}
        ] + conversation_history

        # Get the chatbot's response
        response = conversation.predict(human_input=user_question)
        print("Assistant:", response)

        # Append chatbot response to conversation history
        conversation_history.append({"role": "assistant", "content": response})
      except Exception as e:
        print(f"An error occurred: {e}")
        break

# Testing our Dental clinic booking agent by starting the chat function
chat()
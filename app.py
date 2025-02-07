import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import nltk
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Import stopwords and tokenizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# download nltk data
nltk.download('punkt')
nltk.download('stopwords')


# Cached model loading
@st.cache_resource
def load_chatbot_model():
    try:
        # Use pipeline for easier text generation
        chatbot = pipeline("text-generation", model="distilgpt2")
        return chatbot
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Preprocess user input
def preprocess_input(user_input):
    try:
        stop_words = set(stopwords.words('english'))
        words = word_tokenize(user_input.lower())
        filter_words = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filter_words)
    except Exception as e:
        #st.warning(f"Preprocessing error: {e}")
        return user_input

# Healthcare-specific logic and response generation
def healthcare_chatbot(user_input, chatbot):
    # Preprocess input
    processed_input = preprocess_input(user_input)
    
    # Healthcare-specific keywords and responses
    healthcare_responses = {
        "sneeze": "Frequent sneezing may indicate allergies or a cold. Consult a doctor if symptoms persist.",
        "symptom": "It seems like you're experiencing symptoms. Please consult a doctor for accurate advice.",
        "appointment": "Would you like help scheduling an appointment with a doctor?",
        "medication": "Medication inquiries should be discussed with a healthcare professional.",
        "pain": "Persistent pain should be evaluated by a medical professional.",
    }
    
    # Check for specific healthcare keywords
    for keyword, response in healthcare_responses.items():
        if keyword in processed_input.lower():
            return response
    
    # Fallback to generative model
    try:
        context = """
        This is a healthcare assistant providing general information. 
        Always consult a professional for specific medical advice.
        Common healthcare scenarios include symptoms of cold, flu, and allergies, 
        along with medication guidance and appointment scheduling.
        """
        
        # Generate response using the model
        full_prompt = f"Context: {context}\nQuestion: {processed_input}"
        generated_responses = chatbot(full_prompt, max_length=400, num_return_sequences=1)
        
        # Return the generated response, cleaning up potential duplicates
        return generated_responses[0]['generated_text'].split('Question:')[-1].strip()
    
    except Exception as e:
        st.error(f"Response generation error: {e}")
        return "I'm having trouble generating a response. Could you rephrase your question?"

# Main Streamlit application
def main():
    # Page configuration
    st.set_page_config(
        page_title="Healthcare Assistant ChatBot", 
        page_icon="🏥",
        layout="centered"
    )
    
    # Title and description
    st.title("🏥 Healthcare Assistant ChatBot")
    
    # Load the chatbot model
    chatbot = load_chatbot_model()
    
    # Check if model is loaded
    if chatbot is None:
        st.error("Failed to load the chatbot model. Please check your setup.")
        return
    
    # Chat input
    user_input = st.text_input("How can I assist you today?", key="user_input")
    
    # Submit button
    if st.button("Get Response"):
        if user_input:
            # Display user input
            st.write("You:", user_input)
            
            # Generate and display response
            try:
                response = healthcare_chatbot(user_input, chatbot)
                st.write("Healthcare Assistant:", response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a message to get a response")
    
    # Additional information
    with st.expander("About This Assistant"):
        st.write("""
        ### Features
        - Provides general healthcare information
        - Responds to specific health-related keywords
        - Uses AI to generate contextual responses
        
        ### Limitations
        - Not a substitute for professional medical advice
        - Responses are generated based on available information
        """)

# Run the application
if __name__ == "__main__":
    main()
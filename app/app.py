import streamlit as st
import plotly.graph_objects as go
import pickle
import os
import warnings
from utils import text_transformer

import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Suppress sklearn version warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load models with absolute paths and error handling
try:
    with open(os.path.join(script_dir, "models", "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf_vectorizer = pickle.load(f)
    with open(os.path.join(script_dir, "models", "model.pkl"), "rb") as f:
        model = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# --- Page Configuration ---
st.set_page_config(
    page_title="Spam Detector",
    layout="centered", # or "wide" if you prefer more space
    initial_sidebar_state="collapsed"
)

# --- Custom CSS ---
st.markdown("""
    <style>
        .stButton>button {
            width: 100%;
            border-radius: 20px;
            background-color: #FF4B4B; /* Streamlit's default red for consistency */
            color: white;
            padding: 10px 20px;
            font-size: 1.2rem;
            border: none;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #E63946; /* Slightly darker red on hover */
            color: white;
        }
        .stTextArea [data-baseweb="textarea"] {
            min-height: 150px;
            background-color: #262730; /* Darker background for text area */
            border: 1px solid #4F505B;
            border-radius: 8px;
            padding: 10px;
            color: white;
        }
        .stTextInput label, .stTextArea label, .stFileUploader label {
            font-size: 1.1em;
            color: #ADAFB8;
        }
        .stSuccess {
            background-color: #2E8B57; /* Darker green for success message */
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
        .stError {
            background-color: #B22222; /* Darker red for error message */
            color: white;
            border-radius: 8px;
            padding: 10px;
        }
    </style>
""", unsafe_allow_html=True)


# --- Main Application UI ---
st.title("SMS Spam Detector")
st.write("Enter a message to check if it's spam or legitimate.")

# --- Input Section ---
message_input = st.text_area(
    "Enter your message:",
    height=150,
    placeholder="Paste your message here...",
    key="message_text_area"
)

st.write("---") # Visual separator

# This section is currently unused based on the prompt but kept for future expansion based on image_6d43b8_2.png
# For now, we'll only use the text input and predict button
# st.write("OR")
# uploaded_file = st.file_uploader("(Upload .csv or .zip file only)", type=["csv", "zip"])
# st.write("(Upload .csv or .zip file only)")

# --- Predict Button ---
predict_button = st.button("Predict")

# --- Prediction Logic and Results Display ---
if predict_button:
    if not message_input:
        st.warning("Please enter a message to predict.")
    else:
        #perform operations if text is not empty
        with st.spinner("Analyzing message..."):

            # --- Text Preprocessing ---
            transformed_text = text_transformer(message_input)

            # --- Vectorization ---
            vectorized_text = tfidf_vectorizer.transform([transformed_text])
            
            # --- Model Prediction ---
            #1->spam 0->ham
            result = model.predict(vectorized_text)
            probability = model.predict_proba(vectorized_text)
            if result == 0:
                classification = "Ham"
                confidence_score = round(probability[0][0]*100, 2)
            else:
                classification = "Spam"
                confidence_score = round(probability[0][1]*100, 2)

        st.markdown("---") # Separator before results

        # --- Display Classification Result ---
        if classification == "Ham":
            st.success("Result: This looks like a LEGITIMATE (HAM) message! 🎉")
            gauge_color = "#2E8B57" # Green for Ham
        else:
            st.error("Result: This looks like SPAM message! 🚨")
            gauge_color = "#B22222" # Red for Spam

        st.markdown(f"### Confidence: {confidence_score}%")

        # --- Plotly Gauge Display ---
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence_score,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "", 'font': {'size': 24}}, # No title needed in gauge for clean look
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkgrey"},
                'bar': {'color': gauge_color}, # Dynamic color based on classification
                'bgcolor': "rgba(0,0,0,0)", # Transparent background
                'borderwidth': 0,
                'steps': [
                    {'range': [0, 50], 'color': "rgba(0,0,0,0)"}, # Transparent base
                    {'range': [50, 100], 'color': "rgba(0,0,0,0)"} # Transparent base
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 2},
                    'thickness': 0.75,
                    'value': confidence_score
                }
            }
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", # Transparent background for the entire figure
            font={'color': "white", 'family': "Arial"},
            height=300, # Adjust height as needed
            margin=dict(l=20, r=20, t=20, b=20) # Margins around the gauge
        )

        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
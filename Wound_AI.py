import streamlit as st
from PIL import Image
import tensorflow as tf
from keras.utils import load_img, img_to_array
import numpy as np
from openai import OpenAI
import pandas as pd

# Load the labels from CSV
labels_csv_path = "C:/Users/Rishi/Wound_AI/labels.csv"
labels_df = pd.read_csv(labels_csv_path)
class_labels = labels_df['label'].unique()

# Load the TensorFlow Lite model
tflite_model_path = "C:/Users/Rishi/Wound_AI/model_optimized.tflite"

# Function to load and allocate the TFLite model
def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model(tflite_model_path)

# Get input and output tensor details from the interpreter
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to process the wound image and generate predictions
def get_wound(image_path, interpreter, class_labels):
    img = load_img(image_path, target_size=(224, 224))  # Resize the image to the required size
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array.astype(np.float32) / 255.0  # Normalize the image if required
    
    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], img_array)
    
    # Run inference
    interpreter.invoke()
    
    # Get the predictions from the output tensor
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    predicted_index = np.argmax(predictions)
    predicted_label = class_labels[predicted_index]
    
    # GPT-3.5 integration for initial wound diagnosis and treatment
    client = OpenAI(api_key="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")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a medical assistant that will help diagnose and treat wounds based on its urgency."},
            {"role": "user", "content": f"Give me the best remedies to solve this type of wound and the medical supplies required to treat it: {predicted_label}"}
        ]
    )
    
    return predicted_label, completion.choices[0].message.content

# Function to interact with the chatbot
def chat_with_bot(user_message):
    client = OpenAI(api_key="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")
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a medical assistant that will answer additional questions about the wound diagnosis."},
            {"role": "user", "content": user_message}
        ]
    )
    return completion.choices[0].message.content

# Streamlit App Layout and Aesthetics
st.set_page_config(page_title="AI Wound Diagnosis", layout="centered", initial_sidebar_state="expanded")

# Sidebar Tabs
tab = st.sidebar.radio("Navigate", ["Diagnosis", "About the Product", "Research Paper", "Chatbot"])

# Main Layout and Title for the Diagnosis Tab
if tab == "Diagnosis":
    st.title("AI Wound Diagnosis and Treatment Recommendations")
    st.write("This app uses a deep learning model to diagnose wounds from images and provides GPT-based treatment suggestions. ")
    
    # Columns for better alignment
    col1, col2 = st.columns([2, 1])

    with col1:
        # Image uploader
        uploaded_image = st.file_uploader("Upload a wound image (JPG, PNG, or JPEG)", type=["jpg", "png", "jpeg"])
        
        if uploaded_image is not None:
            # Display uploaded image
            image = Image.open(uploaded_image)
            st.image(image, caption='Uploaded Image', use_column_width=True)
            
            # Save uploaded image to disk temporarily
            image_path = "temp_image.jpg"
            with open(image_path, "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            # Add spinner for loading indication
            with st.spinner("Processing image..."):
                prediction, treatment = get_wound(image_path, interpreter, class_labels)

            # Display results after processing
            st.success(f"**Diagnosis:** {prediction}")
            st.subheader("Recommended Treatment")
            st.write(treatment)

    with col2:
        st.info("Supported Wound Types:")
        for label in class_labels:
            st.write(f"- {label}")
    st.markdown("---")
    st.write("This project was developed by **Rishi** as part of ongoing research in AI-driven medical assistance.")

# About the Product Tab
elif tab == "About the Product":
    st.title("About the Wound Diagnosis Product")
    st.write("""
    ### Purpose of the App
    This app is designed to assist healthcare professionals and individuals in diagnosing various wound types using advanced artificial intelligence. By analyzing images of wounds, the app can provide an accurate diagnosis and offer personalized treatment recommendations. The goal is to help speed up the diagnosis process and improve the overall effectiveness of wound care.

    ### Technology Overview
    The app utilizes state-of-the-art AI technology:
    - A TensorFlow Lite model is used to analyze wound images and classify them into different types.
    - GPT-3.5, a powerful natural language processing model, generates tailored treatment suggestions based on the wound type. It offers recommendations on medical supplies and healing practices.

    ### How It Works
    1. Upload a clear image of the wound.
    2. The app's AI model processes the image to determine the wound type.
    3. Based on the AI diagnosis, GPT-3.5 provides treatment recommendations, including advice on care and required supplies.
    4. Users can interact with the built-in chatbot to ask follow-up questions related to the diagnosis.

    ### Accuracy & Limitations
    This AI model has been trained on a diverse dataset of wound images. While the accuracy is high, the app is meant to complement—**not replace**—professional medical advice. Users should consult a healthcare provider for definitive treatment plans, especially for serious or complex wounds.

    ### Future Development
    Upcoming features will include:
    - Integration with medical databases for more robust diagnosis and recommendations.
    - The ability to track wound healing over time by uploading sequential images.
    - Continued research to improve the model's accuracy and expand the range of supported wound types.
    """)
    st.markdown("---")
    st.write("This project was developed by **Rishi** as part of ongoing research in AI-driven medical assistance.")

# Research Paper Tab
elif tab == "Research Paper":
    st.title("Research Paper")
    st.write("For more detailed information, findings, and methodologies, please refer to the research paper.")
    # Placeholder for shareable link
    st.markdown("[Click here to view the research paper](https://link_to_your_research_paper)")

# Chatbot Tab
elif tab == "Chatbot":
    st.title("Wound Diagnosis Chatbot")
    st.write("Ask more questions about your wound diagnosis and treatment options.")
    
    # Input field for user to ask questions
    user_input = st.text_input("Ask a question about the diagnosis or treatment:")
    
    if st.button("Send"):
        if user_input:
            # Add spinner for loading indication
            with st.spinner("Chatbot is processing..."):
                response = chat_with_bot(user_input)
            
            # Display chatbot response
            st.write(f"**Chatbot:** {response}")
        else:
            st.warning("Please enter a question before clicking send.")

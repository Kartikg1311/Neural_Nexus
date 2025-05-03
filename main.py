import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import pickle
import matplotlib.pyplot as plt
import os
import json
import pandas as pd



st.set_page_config(
    page_title="Sports Image Classifier",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)


def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning(f"CSS file not found: {file_name}")


@st.cache_resource
def load_model_and_classes():
    try:
        model = load_model('models/best_model.keras')
        with open('models/class_indices.pkl', 'rb') as f:
            class_indices = pickle.load(f)
        return model, class_indices
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def predict_image(model, image, class_indices, img_size=224):
    """Make prediction on a single image"""
    try:
        img = image.resize((img_size, img_size))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        pred = model.predict(img_array)
        class_idx = np.argmax(pred)
        predicted_class = list(class_indices.keys())[class_idx]
        predicted_prob = np.max(pred)
        
        return predicted_class, predicted_prob, img
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def show_metrics():
    """Display evaluation metrics if available"""
    try:
        if os.path.exists("models/evaluation/test_accuracy.txt"):
            st.header("Model Performance")
            
            with open("models/evaluation/test_accuracy.txt") as f:
                st.metric("Test Accuracy", f"{float(f.read()):.2%}")
            
            if os.path.exists("models/evaluation/classification_report.json"):
                with open("models/evaluation/classification_report.json") as f:
                    report = json.load(f)
                    st.dataframe(pd.DataFrame(report).transpose())
            
            if os.path.exists("models/evaluation/confusion_matrix.png"):
                st.image("models/evaluation/confusion_matrix.png", use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load metrics: {str(e)}")


def main():

    local_css('app/static/css/styles.css')
    
    model, class_indices = load_model_and_classes()
    
  
    st.sidebar.title("Sports Classifier")

    # try:
    #     st.sidebar.image('app/static/images/logo.png', use_container_width=True)
    # except Exception as e:
    #     st.sidebar.warning(f"Logo not available: {str(e)}")
    
    st.sidebar.markdown("""
    ### About
    This app classifies sports images into 100 different categories using a deep learning model.
    """)
    

    st.title("🏆 Sports Image Classifier")
    st.markdown("Upload an image of a sports activity to classify it")
    
    # show_metrics()
    
    uploaded_file = st.file_uploader(
        "Choose a sports image...", 
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Uploaded Image")
            try:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {str(e)}")
                return
            
        with col2:
            st.subheader("Prediction Result")
            with st.spinner("Classifying..."):
                predicted_class, predicted_prob, processed_img = predict_image(
                    model, image, class_indices
                )
                
                if predicted_class is not None:
                    st.success(f"**Prediction:** {predicted_class}")
                    st.info(f"**Confidence:** {predicted_prob*100:.2f}%")
                    
                    fig, ax = plt.subplots()
                    ax.imshow(processed_img)
                    ax.axis('off')
                    ax.set_title(f"Predicted: {predicted_class} ({predicted_prob*100:.2f}%)")
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.error("Failed to make prediction")

if __name__ == "__main__":
    main()
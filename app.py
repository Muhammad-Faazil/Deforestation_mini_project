import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from functools import lru_cache

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------
# Load the trained model with caching
# ------------------------------------------------------------
@st.cache_resource
def load_cached_model():
    try:
        # Try the fixed model first, then fall back to original
        try:
            model = load_model('deforestation_model_fixed.keras')
            st.sidebar.success("✅ Using improved model")
        except:
            model = load_model('deforestation_model.keras') 
            st.sidebar.info("ℹ️ Using original model")
        
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        logging.error(f"Model loading failed: {e}")
        return None

model = load_cached_model()

# ------------------------------------------------------------
# Improved image preprocessing
# ------------------------------------------------------------
def optimized_preprocess(image):
    # Convert to numpy array once
    img = np.array(image)
    
    # Handle different image formats
    if img.shape[-1] == 4:  # RGBA
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    elif len(img.shape) == 2:  # Grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[-1] == 3:  # RGB
        img = img  # Already in correct format
    
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

# ------------------------------------------------------------
# Safe prediction function
# ------------------------------------------------------------
def safe_predict(model, image):
    try:
        prediction = model.predict(image, verbose=0)
        logging.info(f"Prediction successful: {np.argmax(prediction)}")
        return prediction
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        st.error("Prediction failed. Please try another image.")
        return None

# ------------------------------------------------------------
# Streamlit App
# ------------------------------------------------------------
st.set_page_config(page_title="Deforestation Detection", page_icon="🌲", layout="wide")

st.title("🌲 Deforestation Detection")
st.write("Upload an image to check for deforestation areas.")

# Sidebar for settings
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.5, 0.95, 0.7, 0.05)

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Main content
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", 
                                   type=["jpg", "jpeg", "png"],
                                   help="Upload a satellite or aerial image")
    
    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        # Preprocess and predict
        st.write("Classifying...")
        processed_image = optimized_preprocess(image)
        
        prediction = safe_predict(model, processed_image)
        
        if prediction is not None:
            # Get prediction results
            predicted_class = np.argmax(prediction, axis=1)[0]
            confidence = prediction[0][predicted_class]
            class_labels = ['No Deforestation', 'Deforestation']
            
            # Store in history
            st.session_state.prediction_history.append({
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'prediction': class_labels[predicted_class],
                'confidence': float(confidence),
                'filename': uploaded_file.name
            })
            
            # Display results
            with col2:
                st.subheader("Prediction Results")
                
                if confidence < confidence_threshold:
                    st.warning(f"⚠️ Low confidence prediction ({confidence:.2f}). Please verify manually.")
                else:
                    if predicted_class == 1:
                        st.error(f"🚨 **Prediction:** {class_labels[predicted_class]}")
                    else:
                        st.success(f"✅ **Prediction:** {class_labels[predicted_class]}")
                
                st.write(f"**Confidence:** {confidence:.2f}")
                
                # Confidence visualization
                fig, ax = plt.subplots(figsize=(8, 4))
                classes = ['No Deforestation', 'Deforestation']
                colors = ['green', 'red']
                bars = ax.bar(classes, prediction[0], color=colors, alpha=0.7)
                ax.set_ylabel('Confidence Score')
                ax.set_ylim(0, 1)
                ax.set_title('Prediction Confidence Scores')
                
                # Add value labels on bars
                for bar, value in zip(bars, prediction[0]):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{value:.3f}', ha='center', va='bottom')
                
                st.pyplot(fig)

# History section
st.sidebar.subheader("Prediction History")
if st.session_state.prediction_history:
    history_df = pd.DataFrame(st.session_state.prediction_history)
    st.sidebar.dataframe(history_df.tail(5), use_container_width=True)
    
    # Export functionality
    if st.sidebar.button("Clear History"):
        st.session_state.prediction_history = []
        st.rerun()
    
    if st.sidebar.button("Export Results"):
        csv = history_df.to_csv(index=False)
        st.sidebar.download_button(
            label="Download CSV",
            data=csv,
            file_name="deforestation_predictions.csv",
            mime="text/csv"
        )
else:
    st.sidebar.write("No predictions yet.")

# Add some helpful information
with st.expander("ℹ️ How to use this app"):
    st.write("""
    1. Upload a satellite or aerial image of a forest area
    2. Adjust the confidence threshold in the sidebar if needed
    3. View the prediction results and confidence scores
    4. Check your prediction history in the sidebar
    5. Export results if needed
    """)
    
    st.write("**Tips for better results:**")
    st.write("- Use clear, high-quality images")
    st.write("- Ensure the image shows forest/land areas clearly")
    st.write("- Images should be well-lit and not blurry")
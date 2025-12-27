import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
from desease_detail import disease_info, damage_severity
from isLeaf import is_leaf


# Page configuration
st.set_page_config(
    page_title="AgriScan AI",
    page_icon="🌿",
    layout="wide" 
)

# Load model
model = load_model("models/Groundnut_Model.h5")
class_names = ["Early_Leaf_Spot", "Early_Rust", "Healthy",
               "Late_Leaf_Spot", "Nutrition_Deficiency", "Rust"]


#Set title and description
st.title("🌿 AgriScan AI - Groundnut Leaf Disease Detection",)
st.write("Upload a groundnut leaf image and get prediction.")


#upload section in sidebar
st.sidebar.title("📂 Upload Section")
st.sidebar.caption("*Note: For best results, upload an image containing only one or two groundnut leaves.*")
uploaded_file = st.sidebar.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert("RGB")

    leaf_check = is_leaf(img_path=uploaded_file)  # this returns "Leaf" or "Non Leaf"
    if leaf_check == "Leaf":

        img_resized = img.resize((256, 256))
        img_array = image.img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32')
        
        predictions = model.predict(img_array)
        pred_class = class_names[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        severity = damage_severity.get(pred_class, "⚪ Unknown")


        col1, col2, col3 = st.columns([1,2,1])

        with col1:
            st.image(img, caption="Uploaded Image", width=200)
        with col2:
            st.success(f"**Predicted Disease:** {pred_class}")
            st.info(f"**Confidence:** {confidence:.2f}%")
            st.warning(f"**Estimated Damage Severity:** {severity}")


        #Sisease info section
        info = disease_info[pred_class]

        with st.expander("🌱 Cause",expanded=True):
            st.write(info['cause'])

        with st.expander("💊 Solution",expanded=True):
            st.write(info['solution'])
        # -------------------
        st.markdown("## 🛡️ Prevention Tips")
        st.markdown(f"- {info['prevention']}")
    else:
        st.error("⚠️ This is not a Groundnut leaf image. Please upload a Groundnut leaf image.")


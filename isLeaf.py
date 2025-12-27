from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load your trained model
model = load_model("models/isLeaf_model.h5")
IMG_SIZE = (256, 256)

# to predict leaf or not
def is_leaf(img_path):
    try:
        img = image.load_img(img_path, target_size=IMG_SIZE)
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)[0][0]
        print(f"Raw model output: {pred:.4f}")
        
        return "Non Leaf" if pred >= 0.5 else "Leaf"
    except Exception as e:
        return f"Error: {str(e)}"
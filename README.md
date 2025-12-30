# 🌿 AgriScan AI - Groundnut Leaf Disease Detection

AgriScan AI is a tool that helps farmers find diseases in groundnut plants. You upload a photo of a leaf, and the AI tells you if the plant is sick or healthy.

## 🚀 Live Demo
https://raj-muliyashiya-agriscan-ai.hf.space/

### 🧪 How to Test (Sample Images)
Don't have a groundnut leaf photo handy? I have included a folder named **`test_images`** in this repository for testing purposes.

1. Open the **[test_images](./test_images)** folder.
2. Download any image from the list.
3. Upload it to the **Live Demo** link above to verify the results.

## ✨ How it Works
1. **Upload:** You upload a photo of a groundnut leaf.
2. **Check:** The "Is-Leaf" model checks if the photo is really a leaf.
3. **Scan:** The "Groundnut-Model" finds the disease.
4. **Help:** The app shows the disease name, confidence, and how to fix the problem.

## 📂 Project Structure
- `app.py`: The main website code (Streamlit).
- `isLeaf.py`: Code that checks if the image is a leaf.
- `desease_detail.py`: Information about disease causes and solutions.
- `requirements.txt`: List of libraries needed to run the app.
- `models/`: Folder containing the AI "brain" files (.h5 files).
- `training_scripts/`: Code used to train the AI models.

## 📊 Dataset
I created a custom dataset for this project. 
- **Sources:** Images from Kaggle, Mendeley, and photos I took myself.
- **Classes:** Healthy, Early Leaf Spot, Early Rust, Late Leaf Spot, Nutrition Deficiency, and Rust.

## 🛠️ Built With
- **Python**: The main programming language.
- **TensorFlow/Keras**: Used to build and train the AI models.
- **Streamlit**: Used to create the web interface.
- **Hugging Face**: Used to host the live website.

# 🧬 Skin Cancer Classification

A **Streamlit web app** that classifies skin lesion images as **Benign** or **Malignant** using a Convolutional Neural Network (CNN).

**Features:** Upload JPG/PNG images, predicts lesion type with confidence score, real-time classification.

**Model:** 3-layer CNN with max pooling, dense layers, and dropout; trained on skin lesion dataset; **84.70% test accuracy**; outputs classification report.

**Installation & Usage:**  
1. Clone repo 
2. Install dependencies
3. Run: `streamlit run main.py` and upload an image to see prediction.

**Dependencies:** Python 3.10+, TensorFlow, Keras, Streamlit, NumPy, Pillow, scikit-learn

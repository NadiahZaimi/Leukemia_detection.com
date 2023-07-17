import numpy as np
import PIL.Image
from keras.applications.regnet import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

def load_leukemia_model(model_path):
    model = load_model(model_path)
    return model

def preprocess_image(img_path):
    img = PIL.Image.open(img_path)
    img = img.resize((150, 150))
    img = img.convert('RGB')
    img = np.array(img)
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    return img

def predict_leukemia_type(model, img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    leukemia_type = "Leukemia" if prediction[0][0] > 0.5 else "Normal Blood"  # Assuming binary classification
    return leukemia_type

# Example usage
model_path = 'best_model.h5'  # Path to your trained model file
model = load_leukemia_model(model_path)
img_path = 'path_to_image.jpg'  # Path to the image you want to predict
leukemia_type = predict_leukemia_type(model, img_path)
print("Predicted leukemia type:", leukemia_type)

import os
from flask import Flask, redirect, url_for, send_from_directory, request, render_template
from detectLeukemia import *
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from keras.applications.vgg16 import preprocess_input
from keras.models import load_model
import numpy as np


leukemia_model = load_model('best_model.h5')
leukemia_model.summary()

UPLOAD_FOLDER = 'C:/Users/Nurul Nadiah/OneDrive - Universiti Teknologi MARA/Documents/Test Web/Leukemia-Detection-master/static/dataset'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

def allowed_file(filename):
    for i in range(len(filename)):
        if filename[i] == '.':
            ext = filename[i + 1:]
            break
    if ext in ALLOWED_EXTENSIONS:
        return True

def allowed_file(filename):
    for i in range(len(filename)):
        if filename[i] == '.':
            ext = filename[i+1:]
            break
    if ext in ALLOWED_EXTENSIONS:
        return True

def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.resize((150, 150))
    img = img.convert('RGB')
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_leukemia_type(model, img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    if prediction[0][0] < 0.5:
        leukemia_type = "Abnormal Cells"
        accuracy = 1 - prediction[0][0]
    else:
        leukemia_type = "Normal"
        accuracy = prediction[0][0]
    return leukemia_type, accuracy


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('homepage.html')


@app.route('/detectLeukemia', methods=["GET", "POST"])
@app.route('/detectLeukemia', methods=["GET", "POST"])
def detectLeukemia():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('detectLeukemia.html')

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Perform leukemia prediction
            leukemia_type, accuracy = predict_leukemia_type(leukemia_model, img_path)

            # Load the image using cv2.imread
            a = cv2.imread(img_path)

            # Check if the image was successfully loaded
            if a is None:
                return render_template('detectLeukemia.html')

            # Resize the image
            a = cv2.resize(a, (256, 256))

            # Convert the image to RGB
            img1 = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)

            # Convert the image to grayscale
            img = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)

            # Save the resized and processed images
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'file1.jpg'), a)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'file2.jpg'), img)

            # Enhance contrast
            img_eq = cv2.equalizeHist(img)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'file3.jpg'), img_eq)

            """# Apply adaptive thresholding
            img_thresh = cv2.adaptiveThreshold(img_eq, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'file4.jpg'), img_thresh)"""

            """imgfinal = enhancement(img2, imghist)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'file5.jpg'), imgfinal)"""

            imgthresh = thresholding(img, 150)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'file6.jpg'), imgthresh)

            mask = [[1, 1, 1],
                    [1, 0, 1],
                    [1, 1, 1]]

            erodedimg = erosion(imgthresh, mask)
            openedimg = dilation(erodedimg, mask)

            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'file7.jpg'), openedimg)

            imgEdges = edgeDetection(openedimg)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'file8.jpg'), imgEdges)

            imgcircle, ctr = detectCircles(img, openedimg)
            cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'file9.jpg'), imgcircle)

            return render_template('result.html', cells=ctr, img_path=img_path,
                                   leukemia_prediction=leukemia_type, accuracy=accuracy)

    return render_template('detectLeukemia.html')


if __name__ == "__main__":
    app.debug = True
    app.run()

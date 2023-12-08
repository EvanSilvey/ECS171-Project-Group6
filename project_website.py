from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

classes = ['Black Sea Sprat', 'Gilt Head Bream', 'Horse Mackerel',
           'Red Mullet', 'Red Sea Bream', 'Sea Bass',
           'Shrimp', 'Striped Red Mullet', 'Trout']

cnn_model = load_model('best_CNN.h5')
resnet_model = load_model('best_ResNet50.h5')
vgg16_model = load_model('best_VGG16.h5')

def predict(img_path, model):
    img = image.load_img(img_path, target_size=(400,267))
    img = image.img_to_array(img)
    img = img.reshape(1, 400, 267, 3)
    predicted = model.predict(img)
    prediction = np.argmax(predicted)
    return classes[prediction], predicted

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		model = request.form["selected_model"]
		img = request.files['my_image']

		img_path = "static/" + img.filename
		img.save(img_path)

		prediction = ""
		probabilities = ""

		if model == "CNN":
			prediction, probabilities = predict(img_path, cnn_model)
		elif model == "ResNet50":
			prediction, probabilities = predict(img_path, resnet_model)
		elif model == "VGG16":
			prediction, probabilities = predict(img_path, vgg16_model)

	return render_template("index.html", img_path = img_path, prediction = prediction, probabilities = probabilities, model = model, classes = classes)

if __name__ =='__main__':
	app.run(debug = True)
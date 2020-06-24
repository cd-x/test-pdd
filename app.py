from flask import Flask, render_template, request,url_for,redirect,send_from_directory
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os




#part-prediction (moved from prediction.py)

from tensorflow.keras.models import load_model
import os
import json
import numpy as np
import cv2

import itertools
import random
from collections import Counter
from glob import iglob


file_dir = os.path.dirname(__file__)
MODEL_PATH = os.path.join(file_dir,'model/acc9275own.h5')

model=load_model(MODEL_PATH)


with open(os.path.join(file_dir,'categories.json'), 'r') as f:
    cat_to_name = json.load(f)
    classes = list(cat_to_name.values())
    
#print (classes)

IMAGE_SIZE=(224,224)





def load_image(filename):
    img = cv2.imread(filename)
    #img = cv2.imread(os.path.join(image_dir, filename)) #<-- use in case of test through existing validation dataset
    img = cv2.resize(img, (IMAGE_SIZE[0], IMAGE_SIZE[1]) )
    img = img /255
    
    return img


def predict(image):
    probabilities = model.predict(np.asarray([image]))[0]
    class_idx = np.argmax(probabilities)
    
    return {classes[class_idx]: probabilities[class_idx]}


# def say_hello():
#     print('function added')
#     print(classes)
#     print(model.summary())


#part-predicction


app = Flask(__name__)

root_dir = os.path.dirname(__file__)
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'png'])
app.config['UPLOAD_FOLDER']='uploads'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route("/",methods=['GET'])
def index():
	return render_template('base.html',label='',imagesource='file:://null')


@app.route('/',methods=['GET','POST'])
def upload():
	if request.method == 'POST':
		file = request.files['file']
		#saving file to uploads directory
		file_path=''
		result='please upload a file'
		if file and allowed_file(file.filename):
			filename=secure_filename(file.filename)
			file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			file.save(file_path)

			#prediction part
			img = load_image(file_path)
			result = predict(img)
			#file_path=os.path.join("file:\\",file_path)
	return render_template('base.html',imagesource=file_path,label=result)


#upload API
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


if __name__ == '__main__':
	app.run(debug=False, threaded=False)
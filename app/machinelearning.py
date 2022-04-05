import numpy as np
import cv2
import sklearn
import pickle
import glob
from django.conf import settings 
import os
STATIC_DIR = settings.STATIC_DIR

from keras.models import load_model

classifier = load_model(os.path.join(STATIC_DIR,'models/COVMODEL.h5'))
Detection_dict = {"[0]": "COVID", 
                  "[1]": "Lung_Opacity",
                  "[2]": "Normal",
                  "[3]": "Viral Pneumonia"}
def pipeline_model(path):
    for img in glob.glob(path):
        cv_img = cv2.imread(img)
    #cv2.imshow("ABC",cv_img)
        input_original = cv_img.copy()
        input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)    
        cv_img = cv2.resize(cv_img, (224, 224), interpolation = cv2.INTER_LINEAR)
        cv_img = cv_img / 255.
        cv_img = cv_img.reshape(1,224,224,3) 
        res = np.argmax(classifier.predict(cv_img, 1, verbose = 0), axis=1)
        out=Detection_dict[str(res)]
        print(out)
        return out




























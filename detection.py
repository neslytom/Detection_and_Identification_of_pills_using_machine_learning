import pickle
import numpy as np
import sys
from keras.preprocessing.image import load_img, img_to_array
#from tensorflow.keras.utils import img_to_array,load_img
from keras.models import Sequential, load_model
from keras.preprocessing import image
from keras import backend as K
import cv2

def img_prediction(test_image):

    K.clear_session()
    data = []
    img_path = test_image
    testing_img = cv2.imread(img_path)
    #img_path = "tadalafil 5 MG (18) - Copy.jpg"
    #testing_img = cv2.imread(img_path)
    cv2.imwrite("..\\PillsDetection\static\\pills_detection.jpg", testing_img)
    model = load_model('mobilenet_model.h5')
    test_image = image.load_img(img_path, target_size=(128, 128))
    test_image = image.img_to_array(test_image)
    #print("image2",test_image)
    test_image = np.expand_dims(test_image, axis=0)
    #print("image3",test_image)
    test_image /= 128
    prediction = model.predict(test_image)
    #print("prediction:",prediction)
    lb = pickle.load(open('label_transform.pkl', 'rb'))
    #print("result:",lb.inverse_transform(prediction)[0])
    prediction=lb.inverse_transform(prediction)[0]
    print("pred:",prediction)

    K.clear_session()

    return prediction

#img_prediction()







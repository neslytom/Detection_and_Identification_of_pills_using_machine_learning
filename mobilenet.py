import os
import numpy as np
import pickle
import cv2
from keras.applications import MobileNet
#from tensorflow.python.keras.applications import MobileNet
import keras
from os import listdir
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
# from keras.layers.normalization import BatchNormalization
from tensorflow.keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
# from tensorflow.keras.utils import img_to_array,load_img
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot  as plt
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score, confusion_matrix
# from DBconn import DBConnection
from keras.preprocessing import image
import sys

def build_mobilenet():
    try:
        # database = DBConnection.getConnection()
        # cursor = database.cursor()
        EPOCHS = 10
        INIT_LR = 0.0001
        BS = 4
        default_image_size = tuple((128, 128))
        image_size = 0
        # directory_root = '../input/plantvillage/'
        #width = 225
        #height = 225
        depth = 3
        print("[INFO] Loading Training dataset images...")
        DIRECTORY = "..\PillsDetection\dataset\\train"
        CATEGORIES = [' Amoxicillin 500 MG', 'apixaban 2.5 MG','aprepitant 80 MG','Atomoxetine 25 MG','benzonatate 100 MG','Calcitriol 0.00025 MG','carvedilol 3.125 MG','celecoxib 200 MG',
        'duloxetine 30 MG','eltrombopag 25 MG','montelukast 10 MG','mycophenolate mofetil 250 MG','Oseltamivir 45 MG','pantoprazole40 MG','pitavastatin 1 MG','prasugrel 10 MG','Ramipril 5 MG','saxagliptin 5 MG','sitagliptin 50 MG','tadalafil 5 MG']

        data = []
        clas = []

        for category in CATEGORIES:
            #print("category:",category)
            path = os.path.join(DIRECTORY, category)
            #print("path:",path)
            for img in os.listdir(path):
                img_path = os.path.join(path, img)
                img = image.load_img(img_path, target_size=(128, 128))
                img = img_to_array(img)
                # img = img / 255#
                data.append(img)
                clas.append(category)

        label_binarizer = LabelBinarizer()
        image_labels = label_binarizer.fit_transform(clas)
        pickle.dump(label_binarizer, open('label_transform.pkl', 'wb'))
        n_classes = len(label_binarizer.classes_)
        #print("n_classes:",n_classes)
        np_image_list = np.array(data, dtype=np.float16) / 225.0

        x_train, x_test, y_train, y_test = train_test_split(np_image_list, image_labels, test_size=0.3, random_state=42)
        base_model = MobileNet(include_top=False, input_shape=(128, 128, 3))
        base_model.trainable = False
        classifier = keras.models.Sequential()
        classifier.add(base_model)
        classifier.add(Flatten())
        classifier.add(Dense(20, activation='softmax'))

        opt=Adam(lr=INIT_LR,decay=INIT_LR/EPOCHS)

        classifier.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

        print("[INFO] training network...")

        aug = ImageDataGenerator(
            rotation_range=30, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2,
            zoom_range=0.2, horizontal_flip=True,
            fill_mode="nearest")

        history = classifier.fit_generator(
            aug.flow(x_train, y_train, batch_size=BS),
            validation_data=(x_test, y_test),
            steps_per_epoch=len(x_train) // BS,
            epochs=EPOCHS, verbose=1
        )

        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(acc) + 1)
        # Train and validation accuracy
        plt.plot(epochs, acc, 'b', label='testing accurarcy')
        plt.plot(epochs, val_acc, 'r', label='training accurarcy')
        plt.title('Training and Validation accurarcy')
        plt.legend()
        plt.savefig('static/accuracy.png')

        plt.figure()
        # Train and validation loss
        plt.plot(epochs, loss, 'b', label='testing loss')
        plt.plot(epochs, val_loss, 'r', label='training loss')
        plt.title('testing and training loss')
        plt.legend()
        plt.savefig('static/loss.png')
        plt.show()

        print("[INFO] Calculating build_mobilenet accuracy model accuracy")
        scores = classifier.evaluate(x_test, y_test)
        print("Test Accuracy:", {scores[1] * 100})
        mobilenet_accuracy = scores[1] * 100
        print("accuracy",mobilenet_accuracy)
        print("Training Completed..!")

        # save the model to disk
        # print("[INFO] Saving model...")
        classifier.save('mobilenet_model.h5')
        # sql = "update evaluations set vgg16='"+str(vgg16_accuracy)+"' where sno=1"
        # cursor.execute(sql)
        # database.commit()

    except Exception as e:
        print("Error=", e)
        tb = sys.exc_info()[2]
        print(tb)
        print(tb.tb_lineno)

build_mobilenet()

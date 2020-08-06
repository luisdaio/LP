# import the necessary packages
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.multiclass import OneVsRestClassifier
from PIL import Image
from imutils import paths
from os import path
import os.path
import numpy as np
import argparse
import os
import cv2
import pickle
import mahotas as mt

class ImageClassifier:

    SIZE = tuple([300, 300])

    # extract the haralick texture features function
    def extract_haralick(self, image):

            # calculate haralick texture features for 4 types of adjacency
            textures = mt.features.haralick(image)

            # take the mean of it and return it
            ht_mean = textures.mean(axis=0)
            return ht_mean

    def extract_dataset_features(self):
        # grab all image paths in the input dataset directory, initialize our
        # list of extracted features and corresponding labels
        print("[INFO] extracting image features...")
        imagePaths = paths.list_images("/home/luis/Documents/dataset/seg_train")
        print("[INFO] extracting image features...")
        data = []
        labels = []

        # loop over input images
        for imagePath in imagePaths:
            
            # load the input image from disk, compute color channel
            # statistics, and then update our data list
            # image = Image.open(imagePath)
            print(imagePath)
            image = cv2.imread(imagePath)
            image = cv2.resize(image, self.SIZE)
            image = cv2.GaussianBlur(image, (5,5), cv2.BORDER_DEFAULT)
            image = cv2.Canny(image,100,200)
            features = self.extract_haralick(image)
            data.append(features)
            # extract the class label from the file path and update the
            # labels list
            label = imagePath.split(os.path.sep)[-2]
            labels.append(label)

        # dump features anda labels to disk
        print("[INFO] serializing features...")
        data_ = {"features": data, "labels": labels}
        file_ = open('image_data.pickle', "wb")
        file_.write(pickle.dumps(data_))
        file_.close()
            
    def train_model(self):

        # load the image features from disk
        print("[INFO] loading face embeddings...")
        data = pickle.loads(open('image_data.pickle', "rb").read())    

        # encode the labels, converting them from strings to integers
        print("[INFO] encoding labels...")
        label_encoder = LabelEncoder()
        labels = label_encoder.fit_transform(data["labels"])
        
        # train the model
        print("[INFO] using '{}' model".format("SVM"))
        # model = OneVsRestClassifier(SVC(kernel='linear', probability=True, class_weight='balanced'), n_jobs = -1)
        model = MLPClassifier()
        model.fit(data["features"], labels)

        #save the model to disk
        file  = open('trainned_model_mlp.pickle', 'wb')
        file.write(pickle.dumps(model))
        file.close()

        #save the label encoder to disk
        file2  = open('le.pickle', 'wb')
        file2.write(pickle.dumps(label_encoder))
        file2.close()

    def classify(self, image_path):
        
        # read image
        og_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        og_image = cv2.resize(og_image, self.SIZE)
        blurred_image = cv2.GaussianBlur(og_image, (5,5), cv2.BORDER_DEFAULT)
        edge_image = cv2.Canny(blurred_image,100,200)
        features = self.extract_haralick(edge_image).reshape(1, -1)

        # load trainned model from disk and the label encoder
        model = pickle.loads(open('trainned_model_mlp.pickle', "rb").read())
        label_encoder = pickle.loads(open('le.pickle', "rb").read())

        # make predictions on our data and show a classification report
        print("[INFO] evaluating...")
        predictions = model.predict_proba(features)[0]
        j = np.argmax(predictions)
        s = label_encoder.classes_[j]
        # n = predictions[j]
    
        # put the predicted label in the image
        cv2.putText(og_image, s + " ", (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
        # cv2.imshow("image", og_image)
        # cv2.waitKey()
        
        return og_image

    def write_image(self, name, image, directory_path=""):
        os.chdir(directory_path) 
        cv2.imwrite(name, image)


if __name__ == "__main__":

    im = ImageClassifier()
    # im.extract_dataset_features()
    # im.train_model()
    image = im.classify("/home/luis/Documents/dataset/seg_pred/seg_pred/88.jpg")
    cv2.imshow("image", image)
    cv2.waitKey()
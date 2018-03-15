import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import cnn_model
import cnn_model_batch_normalization as cnn_model_bn
import predict_utils as testing
import confusion_matrix_plot as cm_plot
# Use GPU with theano
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda, floatX=float32"

###### WEIGHTS, uncomment weights for desired model ######
## Dataset 1
WEIGHTS = "weights/BR_CNN_model_DatasetGualandi_v7.2_250.h5"
## Dataset 2
#WEIGHTS = "weights/dataset2_noBatchNormalization_150e.h5"
## Dataset 3
#WEIGHTS = "weights/dataset3_noBatchNormalization_50e.h5"

###### Testing data folders, uncomment desired test ######
## Signer Dependent, Dataset 1 (all angles)
#TESTING_IMG_FOLDER = "dataset/dataset1/testing/"
## Signer Dependent, Dataset 2 (top, bottom, front)
TESTING_IMG_FOLDER = "dataset/dataset2/testing/"
## Signer Dependent, Dataset 3 (front) 
#TESTING_IMG_FOLDER = "dataset/dataset3/testing/"

## Signer Independent, Dataset 1 (all angles)
#TESTING_IMG_FOLDER = "dataset/dataset1-signer-independent-test"
## Signer Independent, Dataset 2 (top, bottom, front)
#TESTING_IMG_FOLDER = "dataset/dataset2-signer-independent-test"
## Signer Independent, Dataset 3 (front)
#TESTING_IMG_FOLDER = "dataset/dataset3-signer-independent-test"

###### Number of testing samples, uncomment desired test ######
#NB_SAMPLES = 1220 # Signer Dependent, Dataset 1
#NB_SAMPLES = 732 # Signer Dependent, Dataset 2
#NB_SAMPLES = 238 # Signer Dependent, Dataset 3
#NB_SAMPLES = 808 # Signer Independent, Dataset 1
#NB_SAMPLES = 474 # Signer Independent, Dataset 2
#NB_SAMPLES = 155 # Signer Independent, Dataset 3

NB_SAMPLES = 2952
# Image dimensions
img_width, img_height = 64, 64
#img_width, img_height = 200, 133
#img_width, img_height = 70, 47

# Sign labels
sign_labels = ["a", "b", "c", "d", "e", "f", "h", "i", "k", "l", "m", "n", "o", "p", "q", "r", "t", "u", "v", "w", "x", "y"]
index_labels = [0, 1, 2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]

# Detecting input shape
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Instantiate model
model = cnn_model.istantiate_model(input_shape, False)
print "INFO: Model instantiated"
# Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam'  ,
              metrics=['accuracy'])
# Load weights
model.load_weights(WEIGHTS)
print "INFO: Model compiled"

# Generate testing data
test_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

testing_generator = test_datagen.flow_from_directory(
    "dataset/dataset1/validation",
    target_size=(img_width, img_height),
    batch_size=64,
    class_mode='categorical',
    shuffle=False)

# CONFUSION MATRIX
Y_pred = model.predict_generator(testing_generator, NB_SAMPLES // 64 + 1  )
y_pred = np.argmax(Y_pred, axis=1)
cm = confusion_matrix(testing_generator.classes, y_pred, index_labels)
print ""
print "CONFUSION MATRIX"
testing.print_cm(cm, sign_labels)
print ""
print ""
print "Classification Report"
print ""
print ""
print classification_report(testing_generator.classes, y_pred, target_names=sign_labels)
print ""
print ""

# PREDICT
print "PREDICTIONS"
print ""
testing.predict_from_folder("dataset/dataset1/validation", img_width, img_height, model, sign_labels, WEIGHTS)


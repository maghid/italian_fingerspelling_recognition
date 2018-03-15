import keras
import os
import numpy
import time
import matplotlib
# Force matplotlib to not use any Xwindows backend. Must be called before pyplot import
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.optimizers import Adam
from hyperdash import Experiment
from keras.callbacks import EarlyStopping
import cnn_model
import cnn_model_batch_normalization as cnn_model_bn

##### Before running check: ########
# DATASET_FOLDER                   #
# nb_train_samples                 #
# test_type                        # 
# epochs                           #
# model                            #
####################################

# Use GPU with theano
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN, device=cuda, floatX=float32"

# Setting non interactive plot
plt.ioff()

# Sign labels
sign_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'h', 'i', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 't', 'u', 'v', 'w', 'x', 'y']

# Dimensions of our images.
img_width, img_height = 64, 64

# Dataset (uncomment desired dataset)
#DATASET_FOLDER = "dataset/dataset1" # all angles
DATASET_FOLDER = "dataset/dataset2" # front, top, bottom
#DATASET_FOLDER = "dataset/dataset3" # front

train_data_dir = DATASET_FOLDER + "/train"
validation_data_dir = DATASET_FOLDER + "/validation"
testing_data_dir = DATASET_FOLDER + "/testing"

# Number of samples (uncomment accordingly to chosen dataset)
#nb_train_samples = 6028 # dataset 1
nb_train_samples = 3484 # dataset 2
#nb_train_samples = 1105 # dataset 3

#nb_validation_samples = 2952 # dataset 1
nb_validation_samples = 1738 # dataset 2
#nb_validation_samples = 519 # dataset 3

#nb_testing_samples = 1220 # dataset 1
nb_testing_samples = 732 # dataset 2
#nb_testing_samples = 238 # dataset 3

epochs = 250
batch_size = 64
#test_type = "_1epochTest_"
test_type = "_noBatchNormalization_"
#test_type = "_3BatchNorm_"

weights_name = "weights/" + DATASET_FOLDER.split("/")[1] + test_type + str(epochs)+ "e" + ".h5" # Location where to save weights

# detecting img data format
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Istantiate and compile model
#model = cnn_model.istantiate_model(input_shape)
model = cnn_model.istantiate_model(input_shape)

model.compile(loss='categorical_crossentropy',
              optimizer='adam'  ,
              metrics=['accuracy'])

# This is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

# This is the augmentation configuration we will use for testing:
test_datagen = ImageDataGenerator(rescale=1. / 255,
    rotation_range=40,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=False)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

testing_generator = test_datagen.flow_from_directory(
    testing_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

# Early Stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=2)

history = model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights(weights_name) # saving weights

# Evaluate model
score = model.evaluate_generator(validation_generator, nb_validation_samples//batch_size, workers=12, use_multiprocessing=False)
print "Evaluate generator results:"
print "Evaluate loss: " + str(score[0])
print "Evaluate accuracy: " + str(score[1])

# Predict model
#scores = model.predict_generator(testing_generator, nb_testing_samples//batch_size, workers=12)
#print "Predict generator: " + scores

# Save loss on txt
loss_history = history.history["loss"]
numpy_loss_history = numpy.array(loss_history)
numpy.savetxt("statistics/"+ DATASET_FOLDER.split("/")[1] + "/loss_history"+ test_type + str(epochs)+ "e" + ".txt", numpy_loss_history*100, delimiter=",")
min_loss = numpy.amin(numpy_loss_history)
print "MIN LOSS: " + str(min_loss)

# Save val_loss on txt
val_loss_history = history.history["val_loss"]
numpy_val_loss_history = numpy.array(val_loss_history)
numpy.savetxt("statistics/"+ DATASET_FOLDER.split("/")[1] + "/val_loss_history"+ test_type + str(epochs)+ "e" + ".txt", numpy_val_loss_history*100, delimiter=",")
min_val_loss = numpy.amin(numpy_val_loss_history)
print "MIN VAL LOSS: " +  str(min_val_loss)

#Save acc on txt
acc_history = history.history["acc"]
numpy_acc_history = numpy.array(acc_history)
numpy.savetxt("statistics/"+ DATASET_FOLDER.split("/")[1] + "/acc_history"+ test_type + str(epochs)+ "e" +".txt", numpy_acc_history*100, delimiter=",")
max_acc = numpy.amax(numpy_acc_history)
print "MAX ACC: " + str(max_acc)

#Save val_acc on txt
val_acc_history = history.history["val_acc"]
numpy_val_acc_history = numpy.array(val_acc_history)
numpy.savetxt("statistics/"+ DATASET_FOLDER.split("/")[1] + "/val_acc_history" + test_type + str(epochs)+ "e" +".txt", numpy_val_acc_history*100, delimiter=",")
max_val_acc = numpy.amax(numpy_val_acc_history)
print "MAX VAL ACC: " + str(max_val_acc)

# Save maximums on txt
maxs = numpy.array([min_loss, min_val_loss, max_acc, max_val_acc])
numpy.savetxt("statistics/" + DATASET_FOLDER.split("/")[1] + "/3BatchNormalization_" + str(epochs)+ "e" + "maximums.txt", maxs, delimiter=",")

# Plotting 
PLOT_FOLDER = "plots/" + DATASET_FOLDER.split("/")[1]

if not os.path.exists(PLOT_FOLDER):
    os.makedirs(PLOT_FOLDER)

print("Saving plots...")

# Val acc and train acc evolution during training for each version
# Summarize history for accuracy
fig1 = plt.figure(1)
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.ylim(0, 1)
plt.xlim(0, epochs)
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

#Save img
fig1.savefig(PLOT_FOLDER + "/accuracy_plot"+ test_type + str(epochs)+ "e" +  ".png")

# Val loss and train loss evolution during training for each version
# Summarize history for loss
fig2 = plt.figure(2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim(0, 1)
plt.xlim(0, epochs)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'], loc='upper left')
#Save img
fig2.savefig(PLOT_FOLDER + "/loss_plot"+ test_type + str(epochs)+ "e" +  ".png")




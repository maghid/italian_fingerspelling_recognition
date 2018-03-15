from PIL import Image
import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

sign_labels_map = {'a': 0, 'c': 2, 'b': 1, 'e': 4, 'd': 3, 'f': 5, 'i': 7, 'h': 6, 'k': 8, 'm': 10, 'l': 9, 'o': 12, 'n': 11, 'q': 14, 'p': 13, 'r': 15, 'u': 17, 't': 16, 'w': 19, 'v': 18, 'y': 21, 'x': 20}

# Predict a single image
def predict_img(img, img_width, img_height, model, weights_name):
    img = img_to_array(img)
    img = img * (1. / 255)
    img = img.reshape((1,) + img.shape)
    pred = model.predict(img)
    predicted_label = np.argmax(pred)
    return predicted_label

# Read and predict all images from a folder
def predict_from_folder(folder, img_width, img_height, model, sign_labels, weights_name):
    images = []
    total_imgs = 0
    correct_predictions = 0
    correct = False
    # Count correct predictions for each angle
    front_correct = 0
    top_correct = 0
    bottom_correct = 0
    left_correct = 0
    right_correct = 0
    # Count total images for each angle
    front_total = 0
    top_total = 0
    bottom_total = 0
    left_total = 0
    right_total = 0
    for dir in os.listdir(folder):
        for filename in os.listdir(folder + "/" + dir):
            correct = False
            img = Image.open(os.path.join(folder + "/" + dir,filename))
            img = np.asarray(img)
            if filename.find('front') != -1:
                front_total += 1
            elif filename.find('top') != -1:
                top_total += 1
            elif filename.find('bottom') != -1:
                bottom_total += 1
            elif filename.find('right') != -1:
                right_total += 1
            elif filename.find('left') != -1:
                left_total += 1
            pred = predict_img(img, img_width, img_height, model, weights_name)
            # Check if prediction is correct
            if str(sign_labels[pred]) == str(filename[0]):
                correct = True
                correct_predictions += 1
                if filename.find('front') != -1:
                    front_correct += 1
                elif filename.find('top') != -1:
                    top_correct += 1
                elif filename.find('bottom') != -1:
                    bottom_correct += 1
                elif filename.find('right') != -1:
                    right_correct += 1
                elif filename.find('left') != -1:
                    left_correct += 1
            total_imgs += 1

    print "Total correct predictions: " + '\n' + " -" + str( (correct_predictions * 100) / total_imgs ) + "% " + " out of " +  str( total_imgs ) + " total images"
    print "Front correct predictions: " + '\n' + " -" + str((front_correct * 100)/ front_total ) + "%" + " out of " + str(front_total) + " total images"
    if top_total != 0:
        print "Top correct predictions: " + '\n' + " -" + str((top_correct * 100)/ top_total ) + "%" + " out of " + str( top_total) + " total images"
    if bottom_total != 0:
        print "Bottom correct predictions:" + '\n' + " -" + str((bottom_correct * 100)/ bottom_total ) + "%" + " out of " + str( bottom_total) + " total images"
    if left_total != 0:
        print "Left correct predictions:" + '\n' + " -" + str((left_correct * 100)/ left_total ) + "%" + " out of " + str( left_total) + " total images"
    if right_total != 0:
        print "Right correct predictions:" + '\n' + " -" + str((right_correct * 100)/ right_total ) + "%" + " out of " + str( right_total) + " total images"

# Function for pretty printing of the confusion matrix
def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    columnwidth = max([len(x) for x in labels]+[5]) # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print "    " + empty_cell,
    for label in labels: 
        print "%{0}s".format(columnwidth) % label,
    print
    # Print rows
    for i, label1 in enumerate(labels):
        print "    %{0}s".format(columnwidth) % label1,
        for j in range(len(labels)): 
            cell = "%{0}d".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print cell,
        print

def print_cm_plot(cm, fileName):
    # PLOTTING CONFUSION MATRIX
    norm_conf = []
    for i in cm:
        a = 0
        tmp_arr = []
        a = sum(i, 0)
        for j in i:
            tmp_arr.append(float(j)/float(a))
        norm_conf.append(tmp_arr)

    fig = plt.figure()
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                    interpolation='nearest')

    width, height = cm.shape

    for x in xrange(width):
        for y in xrange(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), 
                        horizontalalignment='center',
                        verticalalignment='center')

    cb = fig.colorbar(res)
    alphabet = 'ABCDEFHIKLMNOPQRSTUVWXY'
    plt.xticks(range(width), alphabet[:width])
    plt.yticks(range(height), alphabet[:height])
    plt.savefig(fileName)
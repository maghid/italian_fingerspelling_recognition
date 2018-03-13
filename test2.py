import os
import time
print "statistics/"+ os.path.basename(os.getcwd()) + "/loss_history_" + time.strftime("%d-%m-%Y")+"_"+time.strftime("%H:%M:%S")+".txt"


print "statistics/"+ os.path.basename(os.getcwd()) + "/val_acc_history" + "_noBatchNormalization" +".txt"
print "statistics/" + os.path.basename(os.getcwd()) + "/noBatchNormalization_" + "maximums.txt"

DATASET_FOLDER = "dataset/dataset2" # front, top, bottom
print "weights/" + DATASET_FOLDER.split("/")[1] + "_noBatchNormalization" + ".h5"

# Plotting 
PLOT_FOLDER = "plots/" + DATASET_FOLDER.split("/")[1]
epochs = 159

#Save img
print "statistics/"+ DATASET_FOLDER.split("/")[1] + "/loss_history"+ "_noBatchNormalization_" + str(epochs)+ "e" + ".txt"
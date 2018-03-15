# italian_fingerspelling_recognition
Dataset + convolutional neural network for recognizing Italian Sign Language (LIS) fingerspelling gestures.  
CNN with Keras for recognizing letters of the alphabet in LIS (22 classes).   
Fore more info see `documentation.pdf`  

## Folder structure
**dataset\\**  
...... **dataset1\\**   *dataset for training*   
...... **dataset2\\**   *dataset for training*  
...... **dataset3\\**   *dataset for training*  
...... **dataset1-signer-independent-test\\**   *dataset for testing*  
...... **dataset2-signer-independent-test\\**   *dataset for testing*   
...... **dataset3-signer-independent-test\\**   *dataset for testing*   
**plots\\**   *accuracy and loss plots*   
**weights\\**   *weights for each dataset/training*  
**statistics\\**  *txt with statistics about accuracy and loss during training*  
**cnn_model.py**  *base model*  
**cnn_model_batch_normalization.py**  *model with batch normalization*  
**train_cnn.py**  *model training*  
**test_cnn.py**   *model testing (confusion matrix and predict)*  

## Credits
Margherita Donnici & Gianluca Monica

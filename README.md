# Pet_Expression-Specie_CNN

This notebook technically had 2 different Convolution Neural Network (CNN), but it mainly focuses on the pet expression CNN. 

Pet expression dataset: https://www.kaggle.com/datasets/anshtanwar/pets-facial-expression-dataset/data

Assumes the dataset is already in your google drive, the downloaded folder contains multiple subfolder. we are only interested in 4 subfolders ['Sad', 'Angry', 'happy', 'Other']. Data processing is done by going into each folder of interest, and for each file in each folder, a dictionary of 'path' and 'label' is appended into a data list. The data list is then converted into a DataFrame. The DataFrame (data_df) has 2 columns ('path' and 'label'), the path contains the 'path' of each file and the 'label' is the label of the file. 

The label column is replaced with numerical values ('Sad' = 0, 'Angry' = 1, 'happy' = 2, 'Other' = 3) via a label_map function. 

The total dataset contains 1000 samples. Each images needs to be processed into an array. A helpler function is made (processImage(path)) using Image from PIL to open each file path, each image is resize to (250,250) with padding by ImageOps, and then convert to array using tensorflow.keras.preprocessing.image. The arrayis finally normalized to 0 to 1. 

Training and Testing sets are split via train_test_split with test size of 20%, random state set at 45 (not validation set was made). The final training set contained 800 samples, size (250,250), with 3 channels of colors. The Testing set contained 200 samples, size (250,250), with 3 channels of colors. 

# Model 

The model is a sequential model with 3 Conv2D layer with increasing filters (32, 64, 128) of (3,3) follow by MaxPooling2D of (2, 2), all with relu activation. Follow by 2 Dense layer with increasing neurons (64, 128) with relu activation and dropout rate of 0.4. The output layer denses into 4 neurons with softmax activation. The model is compiled with Adam with learning rate of 0.00025, loss of categorical crossentropy. 

Data augmentation is also explored. To save memory, training and test dataframe are split by indices. A training ImageDataGenerator is instanced with rescaling, rotation, wide shifting, height shifting, shearing, zooming, and horizontal flip. The testing ImageDataGenerator is just rescaled. The training data gen flows from the train_df, with x_col 'path' and y_col 'label', target size of (250, 250), a batch size of 32, and it's categorical class (same with testing data gen). 

The model is trained using the train data gen and validation set as test data gen over 380 epochs before the learning rate was adjusted to 0.00025 and then trained for 60 more epochs. 

# Application 

The application loads 2 model (the other model is located in a different notebook), but we will only focus on pet expression model. Loading the model allows the application to reference back to the model. The evaulateMood method takes the image path, converts it to image array via it's private __convert_image__ method. The image array is then passed into the referenced CNN model. The result is then passed into a helper method (model_translate) for presentation. 

The model_translate method takes a model ouput and a labelmap. The model output is normalized to a percentage and a dataframe is made with the label map and each normalized percentage. The highest percentage and index is determined and printed out. 

The 'happyDog' method is very similar to the evulateMood mood. The 'happyDog' method combines evaluateMood and evaluateSpecie methods, 'happy' mood and 'dog'. 

#-----------------------------------------------------------------------------------------------------------------------
# IMPORTING NEEDED PACKAGES
#-----------------------------------------------------------------------------------------------------------------------

import os, random
import string
import shutil
import pandas as pd
from PIL import Image
from keras import layers
from keras import models
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import seaborn as sns
import kaggle
from tqdm import tqdm
import numpy as np
from sklearn.metrics import confusion_matrix
import tarfile

from numpy.random import seed
from tensorflow import random as tfrandom
seed(5)
tfrandom.set_seed(5)

#-----------------------------------------------------------------------------------------------------------------------
# LOADING THE DATA
#-----------------------------------------------------------------------------------------------------------------------

# creating the data folder that will contain all the datasets
# (if these folders already exist, the process of creating them will be skipped
def createdir(mydir):
    try:
        os.mkdir(mydir)
    except OSError:
        pass

# set basedir
basedir = r'C:\Users\TITA\Downloads\data'
#basedir = r'.\data'

# define directories
data1000 = (basedir + r"\data1000")
originaldir = (basedir + r"\original")
destinationdir = data1000
# define directories for the train, val and test splits
train_dir = os.path.join(destinationdir, 'train')
val_dir = os.path.join(destinationdir, 'validation')
test_dir = os.path.join(destinationdir, 'test')
train_red_dir = (basedir + r'\red\train_red')
val_red_dir = (basedir + r'\red\val_red')
test_red_dir = (basedir + r'\red\test_red')

#create all the directories
createdir(r'data')
createdir(data1000)
createdir(train_dir)
createdir(val_dir)
createdir(test_dir)
createdir(basedir + r"\red")
createdir(train_red_dir)
createdir(val_red_dir)
createdir(test_red_dir)


#-----------------------------------------------------------------------------------------------------------------------
# LOADING THE DATA
#-----------------------------------------------------------------------------------------------------------------------


# this code will download the data (it has 2GB). If the data is already downloaded this step can be skipped
kaggle.api.authenticate()
kaggle.api.dataset_download_files('mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out', path=r'data', unzip=True)

os.rename(r".\data\dataset5", r".\data\original")

#-----------------------------------------------------------------------------------------------------------------------
# GET DATA NO KAGGLE
#-----------------------------------------------------------------------------------------------------------------------
import requests

url = "http://www.cvssp.org/FingerSpellingKinect2011/fingerspelling5.tar.bz2"
filename = os.path.join(basedir , url.split("/")[-1])
if os.path.exists(filename):
    pass
else:
    with open(filename, "wb") as f:
        r = requests.get(url)
        f.write(r.content)

tar = tarfile.open(filename, ("r:" + str(filename).split(".")[-1]))
tar.extractall(path=basedir)
tar.close()
os.rename(r".\data\dataset5", r".\data\original")

# optional file delete
if os.path.exists(filename):
    reply = input("Delete original file " + filename + "? [y/[n]] ")
    if reply == 'y':
        os.remove(filename)


#simple
#http://www.cvssp.org/FingerSpellingKinect2011/fingerspelling5.tar.bz2

#hard
#http://www.cvssp.org/FingerSpellingKinect2011/dataset9-depth.tar.gz


#-----------------------------------------------------------------------------------------------------------------------
# PREPROCESS DATA
#-----------------------------------------------------------------------------------------------------------------------

# Alphabet
alphabet_lower = list(string.ascii_lowercase)
alphabet_lower.remove('j')
alphabet_lower.remove('z')

counts = {}
for letter in alphabet_lower:
        counts["count_{0}".format(letter)] = 0

# Rename all images with letter and id (inplace)
for idx, folder in tqdm(enumerate(os.listdir(originaldir))):
    folder_base = os.path.join(originaldir, folder)

    for index, letter in enumerate(os.listdir(folder_base)):
        folder_letter = os.path.join(folder_base, letter)

        for count, image in enumerate(os.listdir(folder_letter)):
            dst = str(letter) + str(counts["count_{0}".format(str(letter))]) + ".png"

            if "depth" in str(image):
                os.remove(os.path.join(folder_letter, image))
            else:
                src = os.path.join(folder_letter, image)
                dst = os.path.join(folder_letter, dst)

                os.rename(src, dst)

                counts["count_{0}".format(str(letter))] += 1


#-----------------------------------------------------------------------------------------------------------------------
# ORGANIZE DATA INTO TRAIN, VALIDATION AND TEST DIRECTORIES
#-----------------------------------------------------------------------------------------------------------------------

# setting the random state for reproducibility
random.seed(20)

# create directories for each class (label)
for dir in [train_dir, val_dir, test_dir]:
    for letter in alphabet_lower:
        createdir(os.path.join(dir, letter))

# Select the number of images for each class in training
# Select images from each person's folder
images_per_person = 280

for idx, folder in tqdm(enumerate(os.listdir(originaldir))):
    folder_base = os.path.join(originaldir, folder)

    for index, letter in enumerate(os.listdir(folder_base)):
        folder_letter_source = os.path.join(folder_base, letter)
        folder_letter_destiny = os.path.join(train_dir, letter)

        images_kept = random.sample(os.listdir(folder_letter_source), k=images_per_person)

        for image in images_kept:
            src = os.path.join(folder_letter_source, image)
            dst = os.path.join(folder_letter_destiny, image)
            shutil.copyfile(src, dst)

# Select images for validation/test from train
images_for_test_val = 200

for index, letter in enumerate(os.listdir(train_dir)):
    folder_letter_source = os.path.join(train_dir, letter)
    folder_letter_destiny_val = os.path.join(val_dir, letter)
    folder_letter_destiny_test = os.path.join(test_dir, letter)

    images_val = random.sample(os.listdir(folder_letter_source), k=images_for_test_val)
    for image in images_val:
        src = os.path.join(folder_letter_source, image)
        dst = os.path.join(folder_letter_destiny_val, image)
        shutil.move(src, dst)

    images_test = random.sample(os.listdir(folder_letter_source), k=images_for_test_val)
    for image in images_test:
        src = os.path.join(folder_letter_source, image)
        dst = os.path.join(folder_letter_destiny_test, image)
        shutil.move(src, dst)


#-----------------------------------------------------------------------------------------------------------------------
# EXPLORE IMAGES' SIZE
#-----------------------------------------------------------------------------------------------------------------------

img_dict = {'filename': [], 'width': [], 'height': []}
for index, letter in enumerate(os.listdir(train_dir)):
    folder_letter = os.path.join(train_dir, letter)
    file_list = [x for x in os.listdir(folder_letter)]
    for img_filename in file_list:
        # opening the image file and getting size info
        img_path = os.path.join(train_dir, letter, img_filename)
        img = Image.open(img_path)
        width, height = img.size
        # adding size parameters and filenames to the dictionary
        img_dict['filename'].append(img_filename)
        img_dict['width'].append(width)
        img_dict['height'].append(height)

# creating a dataframe with the image's sizes
img_df = pd.DataFrame(data=img_dict)
img_df['area'] = img_df['height'] * img_df['width']
img_df['aspect_ratio'] = img_df['width'] / img_df['height']

# plotting the height distribution
plt.hist(img_df['height'], bins=100, color='darkseagreen')
plt.title('Distribution of Images\' Heights')
plt.xlabel('Height of image (in pixels)')
plt.ylabel('Number of images')
plt.show()

# plotting the width distribution
plt.hist(img_df['width'], bins=100, color='darkseagreen')
plt.title('Distribution of Images\' Widths')
plt.xlabel('Width of image (in pixels)')
plt.ylabel('Number of images')
plt.show()

# plotting the width distribution
plt.hist(img_df['area'], bins=100, color='darkseagreen')
plt.title('Distribution of Images\' Areas')
plt.xlabel('Area of image (in pixels)')
plt.ylabel('Number of images')
plt.show()

# plotting the aspect ration distribution
plt.hist(img_df['aspect_ratio'], bins=100, color='darkseagreen')
plt.title('Distribution of Images\' Aspect Ratios')
plt.xlabel('Aspect ratio of image (in pixels)')
plt.ylabel('Number of images')
plt.show()

# distribution of aspect ratio by handshape
img_df['handshape'] = img_df['filename'].apply(lambda x: x[0])
img_df['aspect_ratio'].hist(by=img_df['handshape'])
plt.show()

# descriptive statistics for size
size_desc = img_df.describe()


# joint distribution of height, width
img_df["width_height"] = img_df[["width" , "height"]].apply(lambda row: "_".join(row.values.astype(str)) , axis=1)
img_df["counts_w_l"] = img_df["width_height"].map(img_df["width_height"].value_counts())
plt.scatter(x=img_df["width"], y=img_df["height"], s=img_df["counts_w_l"]*10,  alpha=0.5, c='darkseagreen')
plt.title('Joint Distribution of Height and Width')
plt.xlabel('Width of image (in pixels)')
plt.ylabel('Heigth of image (in pixels)')
# TODO: plot in seaborn with hue by letter
plt.show()

# joint distribution of height, width with hue by letter
sns.scatterplot(x=img_df["width"], y=img_df["height"], hue=img_df["handshape"], palette='muted')
plt.title('Joint Distribution of Height and Width by Letter')
plt.xlabel('Width of image (in pixels)')
plt.ylabel('Heigth of image (in pixels)')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# DATA REDUCTION TO RED CHANNEL
#-----------------------------------------------------------------------------------------------------------------------

# create directories for each class (label)
for letter in alphabet_lower:
    createdir(os.path.join(train_red_dir, letter))
    createdir(os.path.join(val_red_dir, letter))
    createdir(os.path.join(test_red_dir, letter))

# convert all images in train to only use red channel copying them to a new train folder
for index, letter in enumerate(os.listdir(train_dir)):
    folder_letter = os.path.join(train_dir, letter)
    file_list = [x for x in os.listdir(folder_letter)]
    for img_filename in file_list:
        img_path = os.path.join(train_dir, letter, img_filename)
        image = Image.open(img_path)
        r, g, b = image.split()
        r.save(os.path.join(train_red_dir, letter, img_filename))

# convert all images in validation to use red channel copying them to a new validation folder
for index, letter in tqdm(enumerate(os.listdir(val_dir))):
    folder_letter = os.path.join(val_dir, letter)
    file_list = [x for x in os.listdir(folder_letter)]
    for img_filename in file_list:
        img_path = os.path.join(val_dir, letter, img_filename)
        image = Image.open(img_path)
        r, g, b = image.split()
        r.save(os.path.join(val_red_dir, letter, img_filename))

# convert all images in test to use red channel copying them to a new test folder
for index, letter in tqdm(enumerate(os.listdir(test_dir))):
    folder_letter = os.path.join(test_dir, letter)
    file_list = [x for x in os.listdir(folder_letter)]
    for img_filename in file_list:
        img_path = os.path.join(test_dir, letter, img_filename)
        image = Image.open(img_path)
        r, g, b = image.split()
        r.save(os.path.join(test_red_dir, letter, img_filename))

#-----------------------------------------------------------------------------------------------------------------------
# IMAGE DATA GENERATOR - CREATE TENSORS
#-----------------------------------------------------------------------------------------------------------------------

# creates tensors out of the data (normalizing the images)
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_red_dir,
    target_size=(150, 150),  # resizes all images
    batch_size=20,
    class_mode='categorical',
    color_mode='grayscale'
)

validation_generator = val_datagen.flow_from_directory(
    val_red_dir,
    target_size=(150, 150),  # resizes all images
    batch_size=20,
    class_mode='categorical',
    color_mode='grayscale'
)

test_generator = test_datagen.flow_from_directory(
    test_red_dir,
    target_size=(150, 150),  # resizes all images
    batch_size=20,
    class_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)


#-----------------------------------------------------------------------------------------------------------------------
# NETWORK
#-----------------------------------------------------------------------------------------------------------------------

model = models.Sequential()

# feature maps extracted: 100, filter: (3x3), slider: 1
model.add(layers.Conv2D(30, (3, 3), activation='relu', input_shape=(150, 150, 1), padding='same'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(60, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(120, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(120, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())  # vectorize to one dimensional representation
model.add(layers.Dense(24, activation='softmax'))

# model.summary()  # get as the shapes and number of params

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

history = model.fit_generator(train_generator, steps_per_epoch=1200, epochs=5,
                              validation_data=validation_generator, validation_steps=240)

# save the model
id_num = input("Insert GridSearch ID number: ")
model.save_weights('model_weights{}.h5'.format(id_num))
model.save('model_keras{}.h5'.format(id_num))

# apply model to the test set
preds = model.predict(test_generator)
predicted_class_indices = np.argmax(preds, axis=1)
test_labels = test_generator.labels

cm = confusion_matrix(test_labels, predicted_class_indices)
test_score = model.evaluate_generator(test_generator)

def plot_cm(confusion_matrix: np.array, classnames: list):
    """
    Function that creates a confusion matrix plot using the Wikipedia convention for the axis.
    :param confusion_matrix: confusion matrix that will be plotted
    :param classnames: labels of the classes"""

    confusionmatrix = confusion_matrix
    class_names = classnames

    fig, ax = plt.subplots(figsize=(50, 50))
    im = plt.imshow(confusionmatrix, cmap=plt.cm.cividis)
    plt.colorbar()

    # We want to show all ticks...
    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            text = ax.text(j, i, confusionmatrix[i, j],
                           ha="center", va="center", color="w")

    ax.set_title("Confusion Matrix")
    plt.xlabel('Targets')
    plt.ylabel('Predictions')
    plt.ylim(top=len(class_names) - 0.5)  # adjust the top leaving bottom unchanged
    plt.ylim(bottom=-0.5)  # adjust the bottom leaving top unchanged
    return plt.show()

plot_cm(cm, alphabet_lower)

#-----------------------------------------------------------------------------------------------------------------------
# ANALYZING OVERFITTING
#-----------------------------------------------------------------------------------------------------------------------

# DISPLAYING CURVES OF LOSS AND ACCURACY DURING TRAINING
acc = history.history['acc']
val_acc = history.history['val_acc']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
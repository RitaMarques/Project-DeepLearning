import os, random
import string
import shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

#-----------------------------------------------------------------------------------------------------------------------
# PREPROCESS DATA
#-----------------------------------------------------------------------------------------------------------------------
basedir = r'C:\Users\TITA\Downloads\PREPROCESS DATASET'
destinydir = r'C:\Users\TITA\Downloads\IMAGENS_ALL'

# Alphabet
alphabet_upper = list(string.ascii_uppercase)
alphabet_lower = list(string.ascii_lowercase)

counts = {}
for letter in alphabet_lower:
        counts["count_{0}".format(letter)] = 0

# Rename all images with letter and id
for idx, folder in enumerate(os.listdir(basedir)):
    folder_base = os.path.join(basedir, folder)

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


# create directories for the train, val and test splits
train_dir = os.path.join(destinydir, 'train')
os.mkdir(train_dir)
val_dir = os.path.join(destinydir, 'validation')
os.mkdir(val_dir)
test_dir = os.path.join(destinydir, 'test')
os.mkdir(test_dir)

# create directories with classes
alphabet_lower.remove('j')
alphabet_lower.remove('z')
for dir in [train_dir, val_dir, test_dir]:
    for letter in alphabet_lower:
        os.mkdir(os.path.join(dir, letter))

# 350 images to each class in training
# 70 images from each person's folder
images_per_person = 70

for idx, folder in enumerate(os.listdir(basedir)):
    folder_base = os.path.join(basedir, folder)

    for index, letter in enumerate(os.listdir(folder_base)):
        folder_letter_source = os.path.join(folder_base, letter)
        folder_letter_destiny = os.path.join(train_dir, letter)                             # updated to train_dir

        images_kept = random.sample(os.listdir(folder_letter_source), k=images_per_person)
        # images_kept = random.choices(os.listdir(folder_letter_source), k=70)

        for image in images_kept:
            src = os.path.join(folder_letter_source, image)
            dst = os.path.join(folder_letter_destiny, image)
            shutil.copyfile(src, dst)

# move 50 images from train to validation
# move 50 images from train to test
images_for_test = 50

for index, letter in enumerate(os.listdir(train_dir)):
    folder_letter_source = os.path.join(train_dir, letter)
    folder_letter_destiny_val = os.path.join(val_dir, letter)
    folder_letter_destiny_test = os.path.join(test_dir, letter)

    images_val = random.sample(os.listdir(folder_letter_source), k=images_for_test)
    for image in images_val:
        src = os.path.join(folder_letter_source, image)
        dst = os.path.join(folder_letter_destiny_val, image)
        shutil.move(src, dst)

    images_test = random.sample(os.listdir(folder_letter_source), k=images_for_test)
    for image in images_test:
        src = os.path.join(folder_letter_source, image)
        dst = os.path.join(folder_letter_destiny_test, image)
        shutil.move(src, dst)
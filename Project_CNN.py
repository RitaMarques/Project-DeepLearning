import os, random
import string
import shutil
import pandas as pd
from PIL import Image
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# #-----------------------------------------------------------------------------------------------------------------------
# # LOAD THE DATA
# #-----------------------------------------------------------------------------------------------------------------------

# creating the data folder
os.mkdir('data')
os.mkdir(r'data/before')

# this code will download the data (it has 2GB). If the data is already downloaded this step can be skipped
data_url = 'www.cvssp.org/FingerSpellingKinect2011/fingerspelling5.tar.bz2'
import requests
req = requests.get(data_url)
zname = "fingerspelling5.tar.bz2"
zfile = open(zname, 'wb')
zfile.write(req.content)
zfile.close()

shutil.unpack_archive("fingerspelling5.tar.bz2")

req.content
# #-----------------------------------------------------------------------------------------------------------------------
# # PREPROCESS DATA
# #-----------------------------------------------------------------------------------------------------------------------
# basedir = r'.data\PREPROCESS DATASET'
# destinydir = r'.data\IMAGENS_ALL'
#
# # Alphabet
# alphabet_upper = list(string.ascii_uppercase)
# alphabet_lower = list(string.ascii_lowercase)
#
# counts = {}
# for letter in alphabet_lower:
#         counts["count_{0}".format(letter)] = 0
#
# # Rename all images with letter and id
# for idx, folder in enumerate(os.listdir(basedir)):
#     folder_base = os.path.join(basedir, folder)
#
#     for index, letter in enumerate(os.listdir(folder_base)):
#         folder_letter = os.path.join(folder_base, letter)
#
#         for count, image in enumerate(os.listdir(folder_letter)):
#             dst = str(letter) + str(counts["count_{0}".format(str(letter))]) + ".png"
#
#             if "depth" in str(image):
#                 os.remove(os.path.join(folder_letter, image))
#             else:
#                 src = os.path.join(folder_letter, image)
#                 dst = os.path.join(folder_letter, dst)
#
#                 os.rename(src, dst)
#
#                 counts["count_{0}".format(str(letter))] += 1
#
# #-----------------------------------------------------------------------------------------------------------------------
# # ORGANIZE DATA INTO TRAIN, VALIDATION AND TEST DIRECTORIES
# #-----------------------------------------------------------------------------------------------------------------------
#
#
# # create directories for the train, val and test splits
# train_dir = os.path.join(destinydir, 'train')
# os.mkdir(train_dir)
# val_dir = os.path.join(destinydir, 'validation')
# os.mkdir(val_dir)
# test_dir = os.path.join(destinydir, 'test')
# os.mkdir(test_dir)
#
# # create directories with classes
# alphabet_lower.remove('j')
# alphabet_lower.remove('z')
# for dir in [train_dir, val_dir, test_dir]:
#     for letter in alphabet_lower:
#         os.mkdir(os.path.join(dir, letter))
#
# # 350 images to each class in training
# # 70 images from each person's folder
# images_per_person = 70
#
# for idx, folder in enumerate(os.listdir(basedir)):
#     folder_base = os.path.join(basedir, folder)
#
#     for index, letter in enumerate(os.listdir(folder_base)):
#         folder_letter_source = os.path.join(folder_base, letter)
#         folder_letter_destiny = os.path.join(train_dir, letter)                             # updated to train_dir
#
#         images_kept = random.sample(os.listdir(folder_letter_source), k=images_per_person)
#         # images_kept = random.choices(os.listdir(folder_letter_source), k=70)
#
#         for image in images_kept:
#             src = os.path.join(folder_letter_source, image)
#             dst = os.path.join(folder_letter_destiny, image)
#             shutil.copyfile(src, dst)
#
# # move 50 images from train to validation
# # move 50 images from train to test
# images_for_test = 50
#
# for index, letter in enumerate(os.listdir(train_dir)):
#     folder_letter_source = os.path.join(train_dir, letter)
#     folder_letter_destiny_val = os.path.join(val_dir, letter)
#     folder_letter_destiny_test = os.path.join(test_dir, letter)
#
#     images_val = random.sample(os.listdir(folder_letter_source), k=images_for_test)
#     for image in images_val:
#         src = os.path.join(folder_letter_source, image)
#         dst = os.path.join(folder_letter_destiny_val, image)
#         shutil.move(src, dst)
#
#     images_test = random.sample(os.listdir(folder_letter_source), k=images_for_test)
#     for image in images_test:
#         src = os.path.join(folder_letter_source, image)
#         dst = os.path.join(folder_letter_destiny_test, image)
#         shutil.move(src, dst)

#-----------------------------------------------------------------------------------------------------------------------
# SIZE OF THE IMAGES
#-----------------------------------------------------------------------------------------------------------------------

img_dict = {'filename':[], 'width':[], 'height':[]}
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

# creating a dataframe with the image sizes
img_df = pd.DataFrame(data=img_dict)
img_df['area'] = img_df['height'] * img_df['width']
img_df['aspect_ratio'] = img_df['width'] / img_df['height']

# plotting the height distribution
ax = img_df['height'].hist(bins=100)
ax.set_title('Distribution of Images\' Heights')
ax.set_xlabel('Height of image (in pixels)')
ax.set_ylabel('Number of images')
plt.show()

# plotting the width distribution
ax = img_df['width'].hist(bins=100)
ax.set_title('Distribution of Images\' Widths')
ax.set_xlabel('Width of image (in pixels)')
ax.set_ylabel('Number of images')
plt.show()

# plotting the width distribution
ax = img_df['area'].hist(bins=100)
ax.set_title('Distribution of Images\' Areas')
ax.set_xlabel('Area of image (in pixels)')
ax.set_ylabel('Number of images')
plt.show()

# plotting the aspect ration distribution
ax = img_df['aspect_ratio'].hist(bins=100)
ax.set_title('Distribution of Images\' Aspect Ratios')
ax.set_xlabel('Aspect ratio of image')
ax.set_ylabel('Number of images')
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# DATA PRE-PROCESSING
#-----------------------------------------------------------------------------------------------------------------------
train_dir = r'.\data\train'
val_dir = r'.\data\validation'
test_dir = r'.\data\test'

# creates tensors out of the data
train_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(100, 100),  # resizes all images to 150x150
    batch_size=20,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(100, 100),  # resizes all images to 150x150
    batch_size=10,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(100, 100),  # resizes all images to 150x150
    batch_size=10,
    class_mode='categorical'
)


#-----------------------------------------------------------------------------------------------------------------------
# NETWORK
#-----------------------------------------------------------------------------------------------------------------------
model = models.Sequential()

# feature maps extracted: 32   # filter: (3x3)  slider: 1                             (width, height, feature maps)
model.add(layers.Conv2D(100, (3, 3), activation='relu', input_shape=(150, 150, 3)))     # (148, 148,  32)
model.add(layers.MaxPooling2D(2, 2))                                                    # ( 74,  74,  32)
model.add(layers.Conv2D(200, (3, 3), activation='relu'))                                 # ( 72,  72,  64)
model.add(layers.MaxPooling2D(2, 2))                                                    # ( 36,  36,  64)
model.add(layers.Conv2D(400, (3, 3), activation='relu'))                                # ( 34,  34, 128)
model.add(layers.MaxPooling2D(2, 2))                                                    # ( 17,  17, 128)
model.add(layers.Conv2D(400, (3, 3), activation='relu'))                                # ( 15,  15, 128)
model.add(layers.MaxPooling2D(2, 2))                                                    # (  7,   7, 128)
model.add(layers.Flatten())  # vectorize to one dimensional representation              # (6272)
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(400, activation='relu'))                                         # (512) number of neurons
model.add(layers.Dense(24, activation='softmax'))                                        # (1) number of neurons

# model.summary()  # get as the shapes and number of params

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

#-----------------------------------------------------------------------------------------------------------------------
# FITTING THE MODEL
#-----------------------------------------------------------------------------------------------------------------------
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30,
                              validation_data=validation_generator, validation_steps=50)
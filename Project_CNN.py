
# #-----------------------------------------------------------------------------------------------------------------------
# # IMPORTING NEEDED PACKAGES
# #-----------------------------------------------------------------------------------------------------------------------

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
import kaggle
from tqdm import tqdm

# #-----------------------------------------------------------------------------------------------------------------------
# # LOADING THE DATA
# #-----------------------------------------------------------------------------------------------------------------------

def createdir(mydir):
    try:
        os.mkdir(mydir)
    except OSError:
        pass


# creating the data folder and the folder that will contain the original dataset
# (if these folders already exist, the process of creating them will be skipped
createdir(r'data')

# this code will download the data (it has 2GB). If the data is already downloaded this step can be skipped
kaggle.api.authenticate()
kaggle.api.dataset_download_files('mrgeislinger/asl-rgb-depth-fingerspelling-spelling-it-out', path=r'data', unzip=True)

os.rename(r".\data\dataset5" , r".\data\original" )

# #-----------------------------------------------------------------------------------------------------------------------
# # PREPROCESS DATA
# #-----------------------------------------------------------------------------------------------------------------------
#basedir = r'C:\Users\TITA\Downloads\data'
basedir= r'.\data'

data1000 = (basedir + r"\data1000")
createdir(data1000)

originaldir = r'.\data\original'
destinationdir = data1000

# Alphabet

alphabet_lower = list(string.ascii_lowercase)

alphabet_lower.remove('j')
alphabet_lower.remove('z')

counts = {}
for letter in alphabet_lower:
        counts["count_{0}".format(letter)] = 0

# Rename all images with letter and id
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

# create directories for the train, val and test splits
train_dir = os.path.join(destinationdir, 'train')
val_dir = os.path.join(destinationdir, 'validation')
test_dir = os.path.join(destinationdir, 'test')

createdir(train_dir)
createdir(val_dir)
createdir(test_dir)

# create directories with classes
for dir in [train_dir, val_dir, test_dir]:
    for letter in alphabet_lower:
        createdir(os.path.join(dir, letter))

# Select the number of images to each class in training
# Select images from each person's folder
images_per_person = 280

for idx, folder in tqdm(enumerate(os.listdir(originaldir))):
    folder_base = os.path.join(originaldir, folder)

    for index, letter in enumerate(os.listdir(folder_base)):
        folder_letter_source = os.path.join(folder_base, letter)
        folder_letter_destiny = os.path.join(train_dir, letter)                             # updated to train_dir

        images_kept = random.sample(os.listdir(folder_letter_source), k=images_per_person)

        for image in images_kept:
            src = os.path.join(folder_letter_source, image)
            dst = os.path.join(folder_letter_destiny, image)
            shutil.copyfile(src, dst)

# Select images for validation/test

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

# distribution of aspect ratio by handshape
img_df['handshape'] = img_df['filename'].apply(lambda x: x[0])
img_df['aspect_ratio'].hist(by=img_df['handshape'])
plt.show()

# descriptive statistics for size
size_desc = img_df.describe()


# joint distribution of height, width
img_df["width_height"] = img_df[["width" , "height"]].apply(lambda row: "_".join(row.values.astype(str)) , axis=1)
img_df["counts_w_l"] = img_df["width_height"].map(img_df["width_height"].value_counts())
x= img_df["width"]
y= img_df["height"]
z= img_df["counts_w_l"]
plt.scatter(x, y, s=z*10, alpha=0.5, )
# TODO: plot in seaborn with hue by letter
plt.show()

#-----------------------------------------------------------------------------------------------------------------------
# DATA PRE-PROCESSING
#-----------------------------------------------------------------------------------------------------------------------


train_red_dir = (basedir + r'\red\train_red')
val_red_dir = (basedir + r'\red\val_red')
test_red_dir = (basedir + r'\red\test_red')

# TODO: Verificar se posso criar directamente as subdirectorias sem criar a red primeiro
createdir(basedir + r"\red")
createdir(train_red_dir)
createdir(val_red_dir)
createdir(test_red_dir)


for letter in alphabet_lower:
    createdir(os.path.join(train_red_dir, letter))
    createdir(os.path.join(val_red_dir, letter))
    createdir(os.path.join(test_red_dir, letter))


for index, letter in enumerate(os.listdir(train_dir)):
    folder_letter = os.path.join(train_dir, letter)
    file_list = [x for x in os.listdir(folder_letter)]
    for img_filename in file_list:
        img_path = os.path.join(train_dir, letter, img_filename)
        image = Image.open(img_path)
        r,g,b = image.split()
        r.save(os.path.join(train_red_dir, letter, img_filename))

for index, letter in tqdm(enumerate(os.listdir(val_dir))):
    folder_letter = os.path.join(val_dir, letter)
    file_list = [x for x in os.listdir(folder_letter)]
    for img_filename in file_list:
        img_path = os.path.join(val_dir, letter, img_filename)
        image = Image.open(img_path)
        r,g,b = image.split()
        r.save(os.path.join(val_red_dir, letter, img_filename))

for index, letter in tqdm(enumerate(os.listdir(test_dir))):
    folder_letter = os.path.join(test_dir, letter)
    file_list = [x for x in os.listdir(folder_letter)]
    for img_filename in file_list:
        img_path = os.path.join(test_dir, letter, img_filename)
        image = Image.open(img_path)
        r,g,b = image.split()
        r.save(os.path.join(test_red_dir, letter, img_filename))


# creates tensors out of the data (normalizing the images)
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_red_dir,
    target_size=(150, 150),  # resizes all images
    batch_size=50,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_red_dir,
    target_size=(150, 150),  # resizes all images
    batch_size=50,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_red_dir,
    target_size=(150, 150),  # resizes all images
    batch_size=50,
    class_mode='categorical'
)


#-----------------------------------------------------------------------------------------------------------------------
# NETWORK
#-----------------------------------------------------------------------------------------------------------------------

model = models.Sequential()

# feature maps extracted: 32   # filter: (3x3)  slider: 1
model.add(layers.Conv2D(100, (3, 3), activation='relu', input_shape=(150, 150, 3), padding='same'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(200, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(400, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Conv2D(400, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())  # vectorize to one dimensional representation
#model.add(layers.Dropout(0.5))
#model.add(layers.Dense(400, activation='relu'))
model.add(layers.Dense(24, activation='softmax'))

# model.summary()  # get as the shapes and number of params

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])

#-----------------------------------------------------------------------------------------------------------------------
# FITTING THE MODEL
#-----------------------------------------------------------------------------------------------------------------------
history = model.fit_generator(train_generator, steps_per_epoch=100, epochs=30,
                              validation_data=validation_generator, validation_steps=50)
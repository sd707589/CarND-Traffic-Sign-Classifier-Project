import os
import skimage.data
import skimage.transform
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def load_data(data_dir):
    images=[]
    labels=[]
    for f in os.listdir(data_dir):
        shotname,extension = os.path.splitext(f)
        labels.append(int(shotname))
        image_dir=os.path.join(data_dir, f)
        images.append(skimage.data.imread(image_dir))
    return images, labels

ROOT_PATH = os.path.abspath('.')
imgFilePath=os.path.join(ROOT_PATH, "testImg")

images, labels = load_data(imgFilePath)

def display_images_and_labels(images, labels):
    unique_labels = set(labels)
    plt.figure(figsize=(10, 10))
    i = 1
    for label in unique_labels:
        # Pick the first image for each label.
        image = images[labels.index(label)]
        plt.subplot(1, 6, i)  
        plt.axis('off')
        plt.title("Label {0}".format(label))
        i += 1
        plt.imshow(image)
#        plt.subplots_adjust(wspace=0.5)
    plt.show()

# Resize images# Resiz 
images32 = [skimage.transform.resize(image, (32, 32), mode='constant')
                for image in images]
display_images_and_labels(images32, labels)

labels_a = np.array(labels)
images_a = np.array(images32)
print("labels: ", labels_a.shape, "\nimages: ", images_a.shape)
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data
training_file = "train.p"
validation_file="valid.p"
testing_file = "test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

#%%

### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import skimage.data
import skimage.transform
import matplotlib.pyplot as plt
import numpy as np
# TODO: Number of training examples
n_train = len(y_train)

# TODO: Number of validation examples
n_validation = len(y_valid)

# TODO: Number of testing examples.
n_test = len(y_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_train[0].shape

# TODO: How many unique classes/labels there are in the dataset.
n_classes = np.unique(y_train).shape[0]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

#%%
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt

import random
from skimage.transform import SimilarityTransform
from skimage.transform import warp

def distort(img):
    shift_y, shift_x = np.array(img.shape[:2]) / 2.
    
    shift = SimilarityTransform(translation=[-shift_x, -shift_y])
    tf = SimilarityTransform(
        rotation=np.deg2rad(random.uniform(-5.0, 5.0)), 
        scale=random.uniform(0.9, 1.1),
        translation=(random.uniform(-0.1, 0.1)*img.shape[0], random.uniform(-0.1, 0.1)*img.shape[1])
    )
    shift_inv = SimilarityTransform(translation=[shift_x, shift_y])
    
    return warp(img, (shift + (tf + shift_inv)).inverse, mode='edge')

import skimage.exposure
def exposureAdjust(img):
    img_=skimage.exposure.equalize_hist(img)
    return skimage.img_as_ubyte(skimage.exposure.adjust_gamma(img_, random.uniform(0.1,1.01)))

def Preprocess_all(images):
    res = np.empty_like(images)
    for i in range(images.shape[0]):
        res[i] = exposureAdjust(distort(images[i]))
    return res

# Sample images to show
def showDistortEffect(x):
    img_ind=random.randint(0,n_train)
    
    plt.subplot(1,3,1)
    plt.axis('off')
    plt.tight_layout(pad=3.0)
    plt.imshow(x[img_ind])
    plt.title('original_Img')
    plt.subplot(1,3,2)
    plt.axis('off')
    img_ind_dist=distort(x[img_ind])
    plt.axis('off')
    plt.tight_layout(pad=3.0)
    plt.imshow(img_ind_dist)
    plt.title('Distortion')
    img_ind_expos=exposureAdjust(img_ind_dist)
    plt.subplot(1,3,3)
    plt.axis('off')
    plt.tight_layout(pad=3.0)
    plt.imshow(img_ind_expos)
    plt.title('Exposure Adjustment')
    plt.show()
    return

showDistortEffect(X_train)

#%%
# create Artificial data
X_train = np.concatenate((X_train, Preprocess_all(X_train)))
y_train = np.concatenate((y_train, y_train))
print("After adding fake data, X_train's shape is ", X_train.shape[0])

#%%

# draw histogram
def drawHist(y):
    class_index=np.zeros(n_classes)
    for i in range(y.shape[0]):
        class_index[y[i]]+=1
    plt.bar(range(n_classes), list(map(lambda i: class_index[i], range(n_classes))))
    plt.xlabel("Traffic Sign Class")
    plt.ylabel("Count of each sign")
    plt.show()
drawHist(y_train)

# plotting traffic sign images
def display_images_and_labels(_x,y):
    plt.figure(figsize=(15, 15))
    i=1
    if type(y) != list:
        _y=np.unique(y).tolist()
        y_list=y.tolist()
    else:
        _y=y
        y_list=y
    
    for cla in _y:
        image= _x[y_list.index(cla)]
        plt.subplot(7, 7, i)
        plt.axis('off')
        plt.title("Label:{0}-{1}".format(cla, y_list.count(cla)))
        i += 1
        plt.imshow(image)
    plt.show()
display_images_and_labels(X_train, y_train)

#%%

### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.
from sklearn.utils import shuffle
import tensorflow as tf

EPOCHS = 30
BATCH_SIZE = 128
#learning rate
rate = 0.0005
#regularization
lambda1=0.001

# TODO here preprocessing

def convert_grayscale(x):
    conv_gray=tf.constant(1.0/3.0,dtype=tf.float32,shape=(1,1,3,1))
    return tf.nn.conv2d(x, conv_gray,strides=[1,1,1,1],padding='VALID')

def normalize(x):
    var_nor=tf.constant(128.0,dtype=tf.float32)
    return tf.div(tf.sub(x,var_nor), var_nor)
# probability to keep units
keep_prob = tf.placeholder(tf.float32)

#%%

### Define your architecture here.
### Feel free to use as many code cells as needed.
from tensorflow.contrib.layers import flatten
def LeNet(x):    
    mu = 0
    sigma = 0.1

    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))

    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
    
    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1,name='conv1')
    print(conv1)
    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))

    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2,name='conv2')
    print(conv2)
    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(fc1_W))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1, keep_prob)
    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(fc2_W))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)
    fc2 = tf.nn.dropout(fc2, keep_prob)
    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(fc3_W))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, 43)

# use tensorflow to normalize
x_gray= convert_grayscale(x)
x_pre= normalize(x_gray)
logits = LeNet(x_pre)

predicted_labels = tf.argmax(logits, 1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.cast(tf.reduce_mean(cross_entropy),tf.float32)
#tf.add_to_collection("losses",loss_operation)
#loss = tf.add_n(tf.get_collection("losses"))

optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

#%%

### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.
# evaluation
def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

# train session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        train_ac=None
        for offset in range(0, num_examples, BATCH_SIZE):
            global train_ac
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            _,train_ac=sess.run((training_operation,accuracy_operation), 
                                      feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
        
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Train accuracy is {:.3f}".format(train_ac))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
    
    # TODO need add load model
    saver.save(sess, './lenet3')
    print("Model saved")

# evaluation
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    test_accuracy = evaluate(X_test, y_test)
    print("Test Accuracy = {:.3f}".format(test_accuracy))
    
#%%

### Load the images and plot them here.
### Feel free to use as many code cells as needed.
import os

def load_data(data_dir):
    images=[]
    labels=[]
    i=0
    for f in os.listdir(data_dir):
        shotname,extension = os.path.splitext(f)
        labels.append(int(shotname))
        image_dir=os.path.join(data_dir, f)
        temp_img=skimage.transform.resize(skimage.data.imread(image_dir), (32, 32), mode='constant')
        images.append(skimage.img_as_ubyte(temp_img))
        i+=1
    return images, labels

ROOT_PATH = os.path.abspath('.')
imgFilePath=os.path.join(ROOT_PATH, "testImg")

images1, labels1 = load_data(imgFilePath)
display_images_and_labels(images1,labels1)

#%%

### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    predictions=sess.run(predicted_labels, feed_dict={x: images1, keep_prob: 1.0})
print ("Predictions : ",predictions)
print ("Labels      : ",labels1)

#%%

### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.
def calcu_accuracy(labels, answers):
    right_cnt=0.
    for lab, ans in zip(labels , answers):
        if lab==ans:
            right_cnt +=1.
    return right_cnt/len(labels)

print('Accuracy is {:.2%}'.format(calcu_accuracy(labels1, predictions)))

#%%

### Print out the top five softmax probabilities for the predictions on the German traffic sign images found on the web. 
### Feel free to use as many code cells as needed.
softmax_probab=tf.nn.softmax(logits)
top_5=tf.nn.top_k(softmax_probab,5)
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    sf_prob,ans_top_5=sess.run((softmax_probab,top_5), feed_dict={x: images1, keep_prob: 1.0})
ind_top5=ans_top_5.indices
print("The top5 index is:\n",ind_top5)

#%%
def getProbability(index1,all_prob):
    top5_pro=np.zeros_like(index1).astype(np.float64)
    print("top5_pro:{0}, all_prob:{1}".format(top5_pro.dtype, all_prob.dtype))
    for row in range(len(index1)):
        for col in range(len(index1[0])):
            top5_pro[row][col]=all_prob[row][index1[row][col]]
    return top5_pro

probs_top5=getProbability(ind_top5,sf_prob)
np.set_printoptions(precision=4, suppress=True)
print("Corresponding top5 probability is:\n{}".format(probs_top5))

#%%
# plotting traffic sign images
def display_labels_and_ans(imgs,labs,inds,probs):
    plt.figure(figsize=(10, 10))
    show_rows=len(labs)
    show_cols=len(inds[0])
    y_list=y_test.tolist()
    for i in range(show_rows):
        plt.subplot(show_rows,show_cols+1,i*show_cols+i+1)
        plt.axis('off')
        plt.title("Label:{}".format(labs[i]))
        plt.imshow(imgs[i])
        for j in range(show_cols):
            image=X_test[y_list.index(inds[i][j])]
            plt.subplot(show_rows,show_cols+1,i*show_cols+i+2+j)
            plt.axis('off')
#            plt.title("top5_ans:{}".format(inds[i][j]))
            plt.subplots_adjust(wspace=0, hspace=0.3)
            plt.text(0, 0, "AnsLabel: {0}\nProb: {1:.4f}".format(inds[i][j], probs[i][j]))
            plt.imshow(image)
    plt.show()
    return

display_labels_and_ans(images1, labels1,ind_top5,probs_top5)

#%%

### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    plt.suptitle(tf_activation.name)
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", 
                       vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", 
                       vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", 
                       vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", 
                       cmap="gray")
    plt.show()
    return

#%%
def myOutputFeatureMap(imgs,labs):
    indx=random.randint(0,len(labs)-1)
    label=labs[indx]
    img_choose=imgs[indx]
    plt.title("Input img, Label:{}".format(label))
    plt.imshow(img_choose)
    plt.show()
    img=img = np.expand_dims(img_choose, axis=0)

    g1=tf.get_default_graph()
    conv1=tf.Graph.get_tensor_by_name(g1,name="conv1:0")
    conv2=tf.Graph.get_tensor_by_name(g1,name="conv2:0")
    outputFeatureMap(img, conv1)
    outputFeatureMap(img, conv2)
    plt.show()

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    myOutputFeatureMap(images1,labels1)

#%%

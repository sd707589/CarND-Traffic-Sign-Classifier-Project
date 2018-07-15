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

# plotting traffic sign images
def display_images_and_labels(_x,y):
    plt.figure(figsize=(15, 15))
    i=1
    _y=np.unique(y).tolist()
    y_list=y.tolist()
    for cla in _y:
        image= _x[y_list.index(cla)]
        plt.subplot(7, 7, i)
        plt.axis('off')
        plt.title("Label:{0}-{1}".format(cla, y_list.count(cla)))
        i += 1
        plt.imshow(image)
    plt.show()
display_images_and_labels(X_train, y_train)

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
print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
### Data exploration visualization code goes here.
#### Feel free to use as many code cells as needed.
## Visualizations will be shown in the notebook.
##import matplotlib.pyplot as plt
##%matplotlib inline
#### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
#### converting to grayscale, etc.
#### Feel free to use as many code cells as needed.
#from sklearn.utils import shuffle
#import tensorflow as tf
#
#EPOCHS = 20
#BATCH_SIZE = 128
##learning rate
#rate = 0.0005
##regularization
#lambda1=0.001
##import numpy as np
### use numpy
##def np_Preprocess(x):
##    arrayX=np.array(x)
##    arrayX=(arrayX-128)/128
##    return arrayX.tolist()
##
##X_train = np_Preprocess(X_train)
#
## use tensorflow
#def convert_grayscale(x):
#    conv_gray=tf.constant(1.0/3.0,dtype=tf.float32,shape=(1,1,3,1))
#    return tf.nn.conv2d(x, conv_gray,strides=[1,1,1,1],padding='VALID')
#
#def tf_Preprocess(x):
#    var_nor=tf.constant(128.0,dtype=tf.float32)
#    return tf.div(tf.sub(x,var_nor), var_nor)
## probability to keep units
#keep_prob = tf.placeholder(tf.float32) 
#
#from tensorflow.contrib.layers import flatten
#def LeNet(x):    
#    mu = 0
#    sigma = 0.1
#
#    
#    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
#    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 6), mean = mu, stddev = sigma))
#    conv1_b = tf.Variable(tf.zeros(6))
#
#    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b
#    
#    # SOLUTION: Activation.
#    conv1 = tf.nn.relu(conv1)
#
#    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
#    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
#    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
#    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
#    conv2_b = tf.Variable(tf.zeros(16))
#
#    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
#    
#    # SOLUTION: Activation.
#    conv2 = tf.nn.relu(conv2)
#    
#    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
#    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
#    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
#    fc0   = flatten(conv2)
#    
#    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
#    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
#    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(fc1_W))
#    fc1_b = tf.Variable(tf.zeros(120))
#    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
#    
#    # SOLUTION: Activation.
#    fc1    = tf.nn.relu(fc1)
#    fc1 = tf.nn.dropout(fc1, keep_prob)
#    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
#    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
#    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(fc2_W))
#    fc2_b  = tf.Variable(tf.zeros(84))
#    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
#    
#    # SOLUTION: Activation.
#    fc2    = tf.nn.relu(fc2)
#    fc2 = tf.nn.dropout(fc2, keep_prob)
#    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = 43.
#    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
#    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(fc3_W))
#    fc3_b  = tf.Variable(tf.zeros(43))
#    logits = tf.matmul(fc2, fc3_W) + fc3_b
#    
#    return logits
#
#x = tf.placeholder(tf.float32, (None, 32, 32, 3))
#y = tf.placeholder(tf.int32, (None))
#one_hot_y = tf.one_hot(y, 43)
#
## use tensorflow to normalize
#x_gray= convert_grayscale(x)
#x_pre= tf_Preprocess(x_gray)
#logits = LeNet(x_pre)
#predicted_labels = tf.argmax(logits, 1)
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
#loss_operation = tf.cast(tf.reduce_mean(cross_entropy),tf.float32)
##tf.add_to_collection("losses",loss_operation)
##loss = tf.add_n(tf.get_collection("losses"))
#
#optimizer = tf.train.AdamOptimizer(learning_rate = rate)
#training_operation = optimizer.minimize(loss_operation)
#
## evaluation
#correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
#accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#saver = tf.train.Saver()
#
#def evaluate(X_data, y_data):
#    num_examples = len(X_data)
#    total_accuracy = 0
#    sess = tf.get_default_session()
#    for offset in range(0, num_examples, BATCH_SIZE):
#        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
#        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
#        total_accuracy += (accuracy * len(batch_x))
#    return total_accuracy / num_examples
#
## train session
#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    num_examples = len(X_train)
#    
#    print("Training...")
#    print()
#    for i in range(EPOCHS):
#        X_train, y_train = shuffle(X_train, y_train)
#        for offset in range(0, num_examples, BATCH_SIZE):
#            end = offset + BATCH_SIZE
#            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
#            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
#
#        validation_accuracy = evaluate(X_valid, y_valid)
#
#        print("EPOCH {} ...".format(i+1))
#        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
#        print()
#        
#        
#    # TODO need add load model
#    saver.save(sess, './lenet2')
#    print("Model saved")
#
## evaluation
#with tf.Session() as sess:
#    saver.restore(sess, tf.train.latest_checkpoint('.'))
#
#    test_accuracy = evaluate(X_test, y_test)
#    print("Test Accuracy = {:.3f}".format(test_accuracy))
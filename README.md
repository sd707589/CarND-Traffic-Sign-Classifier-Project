# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./image1.png "Distribution"
[image2]: ./image2.png "Brieflook"
[image3]: ./image3.png "FakeImg"
[image4]: ./image4.png "Six traffic Sign"
[image5]: ./image5.png "prediction"
[image6]: ./image6.png "original input"
[image7]: ./image7.png "conv1 Layer weight"
[image8]: ./image8.png "conv2 Layer weight"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](file:///D:/github/repository/CarND-Traffic-Sign-Classifier-Project/report.html)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the **numpy**  library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is [32, 32, 3].
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed.

![alt text][image1]

Then take a brief look at all the traffic sign data.

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to create more fake images with distortion and exposure adjustment. 
Here is an example of an original image and an augmented fake image:

![alt text][image3]

I decided to generate additional data because I want to add grayscale effect and normalization  to all the images of the original and the fake.
To add more data to the the data set, I used the following techniques because I want to generate all the additonal data all together, otherwise I would have to generate during session which obviously cost more time.

``` python
# create Artificial data
X_train = np.concatenate((X_train, Preprocess_all(X_train)))
y_train = np.concatenate((y_train, y_train))
print("After adding fake data, X_train's shape is ", X_train.shape[0])
```

As a second step, I decided to convert the images to grayscale because I want to reduce the interferences from hue.
**From now on, all the preprocess parts were coded in the Tensorflow section without intermediate products -- images**, cause I want to preprocess all the images sent into the tensor graph. To make that goal, I used the following techniques:

``` python
def convert_grayscale(x):
    conv_gray=tf.constant(1.0/3.0,dtype=tf.float32,shape=(1,1,3,1))
    return tf.nn.conv2d(x, conv_gray,strides=[1,1,1,1],padding='VALID')
```

As a last step, I normalized the image data because it benefit the following calculation, and can accelerate the convergence speed. To make that goal, I used the following techniques:

``` python
def normalize(x):
    var_nor=tf.constant(128.0,dtype=tf.float32)
    return tf.div(tf.sub(x,var_nor), var_nor)
```


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 gray image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 14x14x6 					|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling 2x2      	| 2x2 stride,  outputs 5x5x16 					|
| FLATTEN				| outputs 400									|
| Fully connected 		| outputs 120   								|
| DROPOUT				|												|
| Fully connected 		| outputs 84	  								|
| DROPOUT				|												|
| Fully connected 		| outputs 43	 								|
| Softmax				| 		      									|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an Adam optimizer with a learning rate of 0.0005. The batch size is 128. The number of epochs is 30. In addition, a L2 regularization hyperparameter 'lambda1' is provided as 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?  0.883
* validation set accuracy of ?   0.958
* test set accuracy of ?  0.939

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  I chose the original LeNet as the first architecture to try, cause it was already built and could test at once.
* What were some problems with the initial architecture?
  It's valid accuracy was only 0.89 lower than 0.93, which was our goal.
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  In order to avoiding underfitting, I increased the 'epochs' from 10 to 30 to get the weights fully learned, and decreased the 'learning rate' from 0.001 to 0.0005. To get rid of underfitting, I added the 'Dropout' technique in the full connection layers. Furthermore, I wanted to add a L2 regularizer but I failed to load the saver later, so I dismissed it.
* Which parameters were tuned? How were they adjusted and why?
  The variables of weights and biases were tuned. They were adjusted by the back propagation of the cross entropy's gradient descent.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  The convolution layers work well with this problem is because the convolution filters are trainable just like the fully connected layer's filters. A dropout layer helps with creating a successful model is because it forces the network to learn redundant representations, and makes things more robust.

If a well known architecture was chosen:
* What architecture was chosen?  LeNet 
* Why did you believe it would be relevant to the traffic sign application?
  LeNet worked well with the figure's pictures, which the traffic sign pictures are similar to.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
  There is a a 'correct_prediction' variable in the model architecture, which can check whether the index of the biggest member in the final result vector equals that in input label vector. Then analyze the proportion of the equaling amount. And the proportion is just the final model's accuracy. 


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are **six** German traffic signs that I found on the web:

![alt text][image4] 

The fifth image might be difficult to classify because it's too dark and its contrast was bad.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Road narrows on the right | Road narrows on the right |
| Beware of ice/snow | Beware of ice/snow |
| Turn left ahead	| Turn left ahead	|
| Keep left	| Keep left	|
| End of no passing by vehicles over 3.5 metric tons	| End of no passing by vehicles over 3.5 metric tons |
| Speed limit (120km/h)	| Speed limit (120km/h)	|

The model was able to correctly guess all traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.9%. Maybe it's because the distinct quality of the 6 download images were better than that of the test data.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 10-12th cell of the Ipython notebook. Here is a show of the input labels( the first column to the left) and all the predicted labels( probabilities decrease from the left column to the right one).

![alt text][image5] 

For example, for the first image, the model is relatively sure that this is a 'Road narrows on the right' sign (probability of 0.9995), and the image does contain a that sign. The top five soft max probabilities were 0.9995, 0.0005, 0.0000, 0.0000, 0.0000.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
I got the first and the second convolution layers' feature maps which are shown below the original Img.
This is the original input trafiic sign -- 'Keep left' with label 39.

![alt text][image6] 

Belows are the first convolution layer feature maps. From the graphs we can see that the neural network can recognize the edges and the corners of the traffic signs.

![alt text][image7] 

Then let's see the second convolution layer feature map. But they are too odd to understand.

![alt text][image8] 
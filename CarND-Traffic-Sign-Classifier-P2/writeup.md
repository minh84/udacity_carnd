#**Traffic Sign Recognition** 


The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./assets/traffic_sign.png "Visualization Traffic Signs"
[hist_train]: ./assets/hist_train.png "Training Histogram"
[hist_valid]: ./assets/hist_valid.png "Validation Histogram"
[hist_test]:  ./assets/hist_test.png "Test Histogram"
[rgb_vs_y]:   ./assets/rgb_vs_y.png "RBG vs Y"
[gen_data]: ./assets/augmented_data.png "Augmented data"
[model01]: ./assets/model01.png "Model architcture 01"
[model0_overfit]: ./assets/model0_overfit.png "Overfitting Model 0"
[model0_do]: ./assets/model0_do.png "Model 0 with dropout"
[model1_bn]: ./assets/model1_bn.png "Model 1 with BN"
[model2_bn]: ./assets/model2_bn.png "Model 2 with BN"

[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Submission Files

You're reading it! and here is a link to my [submission files]()

* `Traaffic_Sign_Classifier.ipynb` contains all the code for this project which we will walk through in the following sections.
* `writeup.md` is the report that you are reading.

### Data Set Summary & Exploration

#### 1. Basic Summary

After data loaded, I used `numpy` to explore it to answer following questions

* The size of training set is ?
* The size of the validation set is ?
* The size of test set is ?
* The shape of a traffic sign image is ?
* The number of unique classes/labels in the data set is ?

The answer for above questions are given below
```
Number of training examples = 34799
Number of validation examples = 4410
Number of testing examples = 12630
Image data shape = (32, 32, 3)
Number of classes = 43
```

I also check if validation-labels & test-lables are a subset of training-labels:
```
validation-label is subset of train-labels: True
test-label is subset of train-labels: True
```

#### 2. Dataset Visualization

First I visualize the data-images (selected randomly) with their labels and number of samples for training data
<center>

![alt text][image1]

</center>

Next, I visualize the histogram for training/validation and test datasets

<center>

![alt text][hist_train]

![alt text][hist_valid]

![alt text][hist_test]

</center>
We can see that they have similar shape where some labels have a lot of samples while other has quite few samples.

### Design and Test a Model Architecture

Insipered by [baseline model](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), I try to re-produce the result from it. 

#### 1. Pre-processing dataset

##### 1.1 RGB to YUV
As a first step, I decided to convert the images to YUV because

* it allows to use gray-scale image (by using Y channel)
* it also allows to use color (by using UV channel)
* from [wiki](https://en.wikipedia.org/wiki/YUV), YUV encodes image while taking *human perception* into account
* it's used in baseline model which I want to re-produce its result.  

This step is implemented in function `rgb2yuv`
  
Here is an example of a traffic sign image taking only Y channel

<center>

![alt text][rgb_vs_y]

</center>

We can see that the grayscale is easier to visualize than the color one. 

As a last step, following a suggestion in paper [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf), we normalize the inputs by doing

* scale inputs to range [0.0, 1.0] (by dividing to 255.0)
* minus 0.5 to make them contains both positive and negative values

  
##### 1.2 Data augmentation

Looking at available we observe

* dataset is small ~35k for 43 classes
* few classes there is < 200 samples.
  
So we decided to generate additional data using the following techniques (as suggested in baseline model): 

* rotatation with degree randomly selected in [-15.0, 15.0]
* projection which is a combination of shearing + scaling

The implementation uses `skimage.transform` to facilitate the tasks. Let's look at an example of augmented images: 

<center>

![alt text][gen_data]

</center>

The augmented data will help to prevent overfitting and also to make our model genelarize better.  


#### 2. Model architecture

We mimic the baseline model's architecture which described as belows

* **inputs**: are preprocessed images each having shape of 3@32x32 (we use this notation D@WxH to describe shape of tensor), 
however we only use gray-scale input channels since it's more visible (as seen above) and as suggested in the baseline model's paper
* **model-settings**: the following setting can be used to tweak model's structure
    * `state1`: number of filters connected to Y-channels in the first conv-layer   (default = 108)
    * `state2`: number of filters in the second conv-layer (default = 108)
    * `state3`: number of filters in the third conv-layer (default = 0 means that there are 2 conv-layers)
    * `num_units`: number of hidden units in fully-connected layer for the classifier (default = 300)    
    * `use_batchnorm`: flag to decide whether to use batch-normalization or not (see [ref](https://arxiv.org/pdf/1502.03167.pdf)). If `use_batchnorm=True` then we will apply **batch-norm** *before each max-pool layer*

* **first state**:
    * the grayscale Y (1@32x32) input channel is applied conv with kernel-size 5x5 with number of filters = `state1_Y`
    * the UV (2@32x32) input channels are applied conv with kernel-size 5x5 and with number of filters = `state1_UV`
    * we concatenate the two ouputs about to obtain a tensor `state1`@32 x 32 
    * we use ReLU-activation and 2x2 max-pool to create an ouput `state1`@16x16 (named it *conv1*) 
    
* **second state**:
    * output from the **first state** is applied conv with kernel-size 5x5 with number of filters = `state2`
    * then we use ReLU-activation and 2x2 max-pool to create an output `state2`@8x8 (named it *conv2*)
    
* **multi-scale (MS)**  features:
    * we sub-sample *conv1* by using 4x4 max-pool, then flatten it
    * we flatten *conv2*
    * we concatenate the two above to create multi-scale features
    
* **classifier with drop-out**:
    * we use 2-layer (fully connected) with input is the above **(MS)** features 
    * we use number of hidden units = `num_units`
    * we apply **drop-out** on activation-output of the first fully-connected layer
    * the output layer is softmax with cross-entropy loss

The following diagram illustrate the model's architecture for model-setting 
<center>
`state1=108, state2=200, state3=0 num_units=300`

![alt text][model01]

</center>

#### 3. Train, Validate and Test model

To train a model we use AdamOptimizer, we monitor training-loss via `sys.stdout.write` or `tensorboard`. 
Periodically, we measure the prediction-accuracy on sub-sampled of training v.s all validation data. 
This allows us to check whether we are overfitting e.t.c  

We experiment with the following settings

* **training dataset**: use both original dataset and augmented one to check if it improves the performance
* **hyperparameters**:
    * learning rate: 1e-4, 5e-4, 1e-3 
    * epochs: in [10, 50] (we stop if we observe overfitting i.e training-accuracy > 99% while validation-accuracy < 93%)
    * batch_size: 64, 128, 256 
* **improvement technique**:    
    * drop-out to prevent overfiting
    * batch-norm to speed up training

##### 3.1 First trial

We use a similar setting described in the baseline model's paper

|               | model0  |
|:-------------:|:-------:|
| stage1        |  108    |
| stage2        |  108    |
| num_units     |  300    |
| use_batchnorm |  False  |
 
Training above model with the original dataset and the following hyperparameters
 
```
epochs = 15
batch_size = 64
learning_rate = 5e-4
eval_every = 100
keep_prob = 1.0
```

After training, we obtain

|               | train   | validation | test  |
|:-------------:|:-------:|:----------:|:-----:|
| accuracy(%)   |  99.84  |   89.18    | 89.80 |

Looking at the accuracy in function of iterations

<center> 

![alt text][model0_overfit]

</center>

The above curve shows a clear indication of overfitting. To solve overfitting, here we try the following steps

* apply drop-out
* train with more data (augmented one)


##### 3.2 Second trial: using drop-out

We re-train *model0*  with drop-out of `keep_prob=0.5`.
Normally, we would increase number of epochs when using drop-out, however here we have few dataset so we don't increase 
number of epochs (to reduce the risk of overfitting). 

To summarise, the training hyperparameters are

```
epochs = 15
batch_size = 64
learning_rate = 5e-4
eval_every = 200
keep_prob = 0.5
```

After training, we obtain

|               | train   | validation | test  |
|:-------------:|:-------:|:----------:|:-----:|
| accuracy(%)   |  99.97  |   94.56    | 93.14 |

Looking at the accuracy in function of iterations

<center> 

![alt text][model0_do]

</center>

It's clear that drop-out helps us to reduce overfitting hence improve validation-accuracy beyond 93%. 
However it's clear that our model is still overfitting.
 
Now let's try to apply **batch-norm** to see whether if it improves training-speed and accuracy. 

Using same hyperparameters, we obtain

|               | train   | validation | test  |
|:-------------:|:-------:|:----------:|:-----:|
| accuracy(%)   |  99.90  |   96.05    | 94.23 |

<center> 

![alt text][model1_bn]

</center>

It's clear that batch-norm improves not only training-speed but also accuracy. We will always turn on batch-norm from now on.
##### 3.3 Third trial

Let's reduce the capacity of our model by

* only using gray-scale channel
* only using 64 layers for first conv-layer and 128 for the second conv-layer

We have the following settings (called `model2`)

|               | model2  |
|:-------------:|:-------:|
| stage1_Y      |  64     |
| stage1_UV     |  0      |
| stage2        |  128    |
| num_units     |  200    |
| use_batchnorm |  False  |
| l2_reg        |  1e-8   |

Using the same hyperparameters as before
```
epochs = 15
batch_size = 64
learning_rate = 5e-4
eval_every = 200
keep_prob = 0.4
```

We obtain

|               | train   | validation | test  |
|:-------------:|:-------:|:----------:|:-----:|
| accuracy(%)   |  99.83  |   94.29    | 94.22 |

with following training/validation accuracy curves

<center> 

![alt text][model2_bn]

</center>

Reducing capacity doesn't help when training with original data.

##### 3.4 Fourth trail: train with augmented data



####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?



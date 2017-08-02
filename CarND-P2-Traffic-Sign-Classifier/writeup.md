#**Traffic Sign Recognition** 


The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[training_input]: ./assets/traffic_sign.png "Visualization Traffic Signs"
[hist_train]: ./assets/hist_train.png "Training Histogram"
[hist_valid]: ./assets/hist_valid.png "Validation Histogram"
[hist_test]:  ./assets/hist_test.png "Test Histogram"
[jittered_examples]: ./assets/jittered_examples.png "Augmented data"
[rgb_vs_y]:   ./assets/rgb_vs_y.png "RBG vs Y"
[model01_lenet5]: ./assets/model01_lenet5.png         "Model architcture 01"
[model02_multiscale]: ./assets/model02_multiscale.png "Model architcture 02"

[exp01_rgb]: ./assets/exp01_rgb.png "Experiment 01 RGB"
[exp01_y]: ./assets/exp01_y.png "Experiment 01 Y"

[exp02_rgb_scaled]: ./assets/exp02_rgb_scaled.png "Experiment 02 RGB scaled"
[exp02_rgb_norm]:   ./assets/exp02_rgb_norm.png "Experiment 02 RGB normalized"

[exp03_dropout]: ./assets/exp03_dropout.png "Experiment 03 Dropout layer"

[exp04_aug5x]: ./assets/exp04_aug5x.png "Experiment 04 nn2 + aug_5x"
[exp04_aug3k]: ./assets/exp04_aug3k.png "Experiment 04 nn2 + aug_3k"

[exp05_aug5x_100e]: ./assets/exp05_aug5x_100e.png "Experiment 05 100 epochs"
[exp06_aug5x_100e_lr5e-4]: ./assets/exp06_aug5x_100e_lr5e-4.png "Experiment 06 100 epochs lr=5e-4"

[exp07_ms_rgb]: ./assets/exp07_ms_rgb.png "Experiment 07 Multi-scale RGB"
[exp07_ms_y]: ./assets/exp07_ms_y.png "Experiment 07 Multi-scale Y"

[exp08_early_stopping]: ./assets/exp08_nn3_early_stopping.png "Early stopping"

[test_images]:        ./assets/test_images.png        "Test Images"
[test_images_scaled]: ./assets/test_images_scaled.png "Test images scaled"
[top_5]: ./assets/top_5.png "Top 5 Predictions"

[vis_con_01]: ./assets/vis_con_01.png "Visualize Conv 01"
[vis_con_02]: ./assets/vis_con_02.png "Visualize Conv 02"

[vis_inputs]: ./assets/vis_inputs.png "Visualize Conv 01"
[vis_input0]: ./assets/vis_input0.png "Visualize Conv 02"
[vis_input1]: ./assets/vis_input1.png "Visualize Conv 01"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Submission Files

You're reading it! and here is a link to my [submission files]()

* `Traffic_Sign_Classifier.ipynb` contains all the code for this project which we will walk through in the following sections.
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

![alt text][training_input]

</center>

Next, I visualize the histogram for training/validation and test datasets

<center>

![alt text][hist_train]

![alt text][hist_valid]

![alt text][hist_test]

</center>
We can see that they have similar shape where some labels have a lot of samples while other has quite few samples.

### Design and Test a Model Architecture

In the project it suggests to use LeNet5 as a start point and it also introduces the [baseline model](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). We will use both (with some modifications) in this project. 

#### 1. Pre-processing dataset

##### 1.1 Data augmentation

Looking at available we observe

* dataset is small ~35k for 43 classes
* few classes there is < 200 samples.
  
So we decided to generate additional data using the following techniques (as suggested in baseline model): 

* rotatation with degree randomly selected in [-10.0, 10.0]
* projection which is a combination of shearing + scaling

An jittered image is created by a rotation following by a projection. Note that we need to scale it back into `uint8` as original input.

The implementation uses `skimage.transform` to facilitate the tasks. Let's look at an example of augmented images: 

<center>

![alt text][jittered_examples]

</center>

The augmented data will help to prevent overfitting and also to make our model genelarize better.  

##### 1.2 Preprocessing
In this section, there are a lot of choice for example

* RGB or grayscale (from now on **Y** mean grayscale)
* normalized input or not e.t.c

We limit to try the following test cases
1. use RGB and scale it to [0, 1]
2. use Y and scale it to [0,1]
3. use RGB and normalized it to have mean 0 and stddev 1.
4. use Y and normalized it

Let's look some examples RGB v.s Y
<center>

![alt text][rgb_vs_y]

</center>

We can see that the grayscale ones are easier to visualize than the color ones since the color ones are too dark.  

#### 2. Model architecture
We will try two architectures here
* **LeNet5** as suggested in the project, this has the following architecture 
<center>

![alt text][model01_lenet5]

</center>

* **Multi-scale** as described in the baseline model, this has the following architecture

<center>

![alt text][model02_multiscale]

</center>

We do the following changes to the original models
 * we use *ReLu* as activation
 * for LeNet5 we use 2 conv-net layers with 8-16 filters each, the classifier is the same 

#### 3. Train, Validate and Test model

To train a model, we use `tf.AdamOptimizer` with the following default hyperparameters

|               | default  |
|--------------:|:--------:|
| learning_rate |  1e-3    |
| batch_size    |  64      |

To facilitate the training/evaluation we implement the following class
* `NeuralNetwork`: in `nn.py` allows us to construct a net in chaining rule (inspired by `Keras`)
* `Dataset`: in `data.py` allows us to transform data, generate mini-batches
* `TrainingSession`: in `train.py` allows us to create/save/load a `tf.Session`

Now, let's train a model for German Traffic Sign recognition 

##### Experiment 1: LeNet5 RGB of Y

In the baseline model, it suggests grayscale performs better. However LeNet5 has different architecture, so first 
let check whether RGB or Y is better?
 
We train with original data with 20 epochs

<center>

![alt text][exp01_rgb]

![alt text][exp01_y]

</center>

Looking at above result, we observe 

* for all data-pipelines, our model is well overfitting
* using only Y-channel is slightly less accuracy than RGB

Since using RGB gives slightly better accuracy, we will keep using RGB for LeNet5 (note this is against what stated in 
the baseline-model papers but here we use different net's architecture, we will test Y-channel later with 
multi-scale architecture).

##### Experiment 2: normalized input?
 
As suggested in the paper [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf) by Yann LeCunn et al. 
Let's test whether if normalized input helps

<center>

![alt text][exp02_rgb_scaled]

![alt text][exp02_rgb_norm]

</center>

Looking at above result, it's hard to decide whether we should use normalized inputs or not since validation-accuracy are similar (both are still very overfitting). However we notice that training-accuracy with normalized inputs converge quicker. So we will use normalized input from now on.
Now let's tackle the over-fitting issue. One can do the following steps
* use dropout layer
* train with more data: the augmented one

##### Experiment 3: Dropout layer

Let's add Dropout layer with keep-prob 0.5 after each fully-connected layer, we obtain the following result after 15 epochs

<center>

![alt text][exp03_dropout]

</center>

Our model finally passed the 93%, but it's clear that it still overfitting. The accuracy after training is given as belows 


|          | train   | validation | test  |
|:--------:|:-------:|:----------:|:-----:|
| original |  99.22  |   93.88    | 93.67 |

##### Experiment 4: Train with augmented data

The next step to solve the overfitting issue is to train with more data. Fortunately, we have generated augmented datas
* aug_5x: is the orginal data + 4x jittered images (we generate 4 jittered images from the original one)
* aug_3k: is the original data + random jittered images to make each label having 3k samples

Using augmented data we obtain the following result 

<center>

![alt text][exp04_aug5x]

![alt text][exp04_aug3k]

</center>

Oberserving above result, we can see that drop-out + augmented data is very effective at preventing overfitting the learning curve for training and validation accuracy are closer. The accuracy after training is given as belows

|                 | train   | validation | test  |
|:---------------:|:-------:|:----------:|:-----:|
| augmentation 5x |  94.95  |   94.33    | 94.18 |
| augmentation 3k |  96.02  |   94.22    | 92.82 |

Note that the augmentation 3k doesn't work well since its distribution is different from the validation set and the test set. From now on we only use augmentation 5x for this project.

The next step is to consider the following questions
* Has the learning curve converged?
* Is the learning rate too high, too low?

##### Experiment 5: Train with more epochs

Looking at the above learning curves, it seems that it hasn't converged yet. Let's try with more epochs, here is the results after training with 100 epochs

<center>

![alt_text][exp05_aug5x_100e]

</center>

With 100 epochs, both training and validation accuracy are improved

|                 | train   | validation | test  |
|:---------------:|:-------:|:----------:|:-----:|
| augmentation 5x |  97.41  |   97.35    | 94.62 | 


##### Experiment 6: Train with lower learning rate

Looking at the above learning curves, the training accuracy is fluctuating and not improving after 40 epochs. 
One could try to train with lower learning rate

<center>

![alt_text][exp06_aug5x_100e_lr5e-4]

</center>

We observe that not only the learning-curves are smoother and we obtain a better training accuracy, 
however validation accuracy is worse

|                 | train   | validation | test  |
|:---------------:|:-------:|:----------:|:-----:|
| augmentation 5x |  98.22  |   95.96    | 95.35 |

We think this might be the best that the current model can do. 
We might need other architecture to improve the accuracy further.

##### Experiment 7: Try multi-scale architecture

Since the baseline model gives a very high accuracy > 98%, it might be worth to try it out. Let's use

* 2 layer conv-net 108-108 filters each
* multi-scale features: concatenate features from the first conv-layer (after max-pool 4x4) and the second one
* we use only one fc-layer of 100 hidden units as suggested in the baseline model

The training parameters are

<center>`learning_rate=5e-4, batch_size=128 and epochs=100`</center>

We obtain the following result

 
<center>

![alt text][exp07_ms_rgb]

![alt text][exp07_ms_y]

</center>

The accuracy table is given as belows

|     | train   | validation | test  |
|:---:|:-------:|:----------:|:-----:|
| RGB |  99.94  |   96.92    | 96.79 |
| Y   |  99.98  |   98.19    | 97.19 |

This multi-scale architecture works well with grayscale image, we finally obtain a validation accuracy > 98%.

##### Experiment 8: Try early stopping

From observing the learning curve, we notice that validation accuracy can archive its maximum before all training epochs are done. 
There is a technique called `early-stopping` that suggests to use this state instead of the final state. 

We try this with multi-scale model and grayscale images:

<center>

![alt text][exp08_early_stopping]

</center>

The early stopping have the following accuracy

|     | train   | validation | test  |
|:---:|:-------:|:----------:|:-----:|
| Y   |  99.66  |   98.32    | 96.24 |

We will use this as our final model since it's validation accuracy is highest.



####4. Model selection

Going through a series of experiments, we choose the final model to be the multi-scale model
with following architecture
* 108-108 conv-net 
* one fc with 100 hidden units
* dropout with keep_prob=0.5

The training set is the gray-scale images of augmented data 5x. 
 
My final model results were:

|                                        | train   | validation | test  |
|---------------------------------------:|:-------:|:----------:|:-----:|
| multi-scale 108-108 fc 100 dropout 0.5 |  99.66  |   98.32    | 96.24 |

Go through the experiments we learnt
* RGB or Y depends on neural net's architecture, when the net is fairly small RGB performs slightly better than Y.
However when the net is big, RGB tends to overfit quicker.
* Deep neural net can easily overfit training data
* To solve overfitting: combination of dropout + augmented data is very effective
* when the learning-curve is fluctuating, reducing learning rate helps

There are a lot more to be experiment with
* Tunning batch-size (we notice that small batch-size converge quicker in term of number of epochs but runs slower in term of number seconds per epoch)
* Tunning learning-rate by using random learning-rate
* Try other model parameter e.g number of layers for conv-net, number of hidden-unit for fully-connected layers
* Try batch-normalization layer

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

<center>

![alt text][test_images]

</center>

We can see that images comes in different shape and quality, since our net only accept 32x32 we need to rescale it
<center>

![alt text][test_images_scaled]

</center>

After images resized, even human might find it's difficult to classify some of the above images.

####2. Here is prediction result along with top-k probability
Here is the table of true labels v.s predictions

|  image      |           label        | prediction              |
|------------:|:----------------------:|:-----------------------:|
| test_01.jpg |   Speed limit (30km/h) |    Speed limit (30km/h) |
| test_02.jpg |   Stop                 |    Stop                 |
| test_03.jpg |   Children crossing    |    Slippery road        |
| test_04.jpg |   Prioriy road         |    Prioriy road         |
| test_05.jpg |   Right-of-way...      |    Right-of-way...      |
| test_06.jpg |   General caution      |    General caution      |
| test_07.jpg |   Yield                |    Yield                |
| test_08.jpg |   Keep right           |    Keep right           |
| test_09.jpg |   Roundabout...        |    Roundabout...        |
| test_10.jpg |   Pedestrians          |    Pedestrians          |

The prediction accuracy is 90% with the failed case is `test_03.jpg`, but we can see the resized `test_03.jpg` looks like a slippery road so it's understandable that our net made mistake here.

Let's the top-5 softmax probabilities

<center>

![alt text][top_5]

</center>

The net is very confident on 9/10 images except for the `test_03.jpg`. We can see that the net is confused since the resized image look similar to all top-5 labels.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Let's visualize conv-net layer for LeNet5 architecture


<center>

![alt text][vis_con_01]
![alt text][vis_con_02]

</center>

We notice that the `FeatureMap 4` of the first layer seem to be activated via shape, the second one is not easy to interpretable. Now let's look an image that has a lot of traffic sign in it and an image that has a dog

<center>

![alt text][vis_inputs]
![alt text][vis_input0]
![alt text][vis_input1]

</center>

The first layer looks more noisy than before but looking at `FeatureMap 4`, it still looks like shape of the front image.

In the scope of this project, we won't look further on visualizing deep layers but if you are interested there are few interesting papers on this
* Understanding Neural Networks Through Deep Visualization by [Yosinski et al.](http://www.evolvingai.org/files/2015_Yosinski_ICML.pdf)
* Visualizing and Understanding Convolutional Networks by [Zeiler  et al.](https://www.cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf)
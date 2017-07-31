#**Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[steering_hist]: ./assets/steering_hist.png "Steering histogram"
[combined_hist]: ./assets/combined_hist.png "Combined steering histogram"

[img_flipped]: ./assets/img_flipped.png "Flipped image"
[img_shadowed]: ./assets/img_shadowed.png "Shadowed image"
[img_brightness]: ./assets/img_brightness.png "Brightness image"
[img_translated]: ./assets/img_translated.png "Translated image"
[img_cropped]: ./assets/img_cropped.png "Cropped image"
[img_preprocessed]: ./assets/img_preprocessed.png "Preprocessed image"

[multi_camera]: ./assets/carnd-using-multiple-cameras.png "Multi camera"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.

### Submission Files

The project source code is available [here](https://github.com/minh84/udacity_carnd/tree/master/CarND-Behavioral-Cloning-P3) which contains

* `writeup.md` is the report that you are reading
* `model.py` is the script to create/train model
* `drive.py` is the script to drive the car in Autonomous mode
* `utils.py` contains some useful function for data-augmentation and data-preprocessing
* `model.h5` is trained Keras model
* `video_track1.mp4` is a video recording the car driving autonomously around the track 1
 

### Data

Udacity kindly provides a [sample training data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) however the dataset is a bit small and not uniform (containing a lot of zero)

<center>

![alt text][steering_hist]

</center>

The data is a bit skewed (more positive than negative)
```
min angle -0.9426954
max angle 1.0
number of positive 1900
number of negative 1775
number of samples 8036
```

#### Collecting additional data
This is the most challenging task for me in this project since I can't control the car to drive well in the simulator using either keyboard or mouse. 

Since my driving data is bad, collecting more data doesn't help. 
Fortunately, looking from slack-forum I found this [data](https://nd013.s3.amazonaws.com/track1.zip).

While using the simulator to collecting data, I would suggest the following improvement to make collecting data more easily
* allow to **set fixed speed** in Training mode so that we can control only steering angle

The combined dataset has following distribution & stat

<center>

![alt text][combined_hist]

</center>

```
min angle -1.0
max angle 1.0
number of positive 5027
number of negative 4536
number of samples 17368
```

#### Data augmentation
Since the dataset is small, it's neccesary to do data-augmentation to prevent overfitting and improve the model's generalization. 

As suggested in the project page and also looking on the internet, we will use the following technique to augment data

##### Randomly flip the image
Using `np.fliplr`, we not only double the data but also make positive/negative steering to balance out. 

Here is an example of flipped image

<center>

![alt text][img_flipped]

</center> 

This is implemented in `utils.random_flip`.

##### Randomly add a shadow to the image
We generate two points one on top and one in the bottom, then we draw a line between two points to divide the image into two parts. 
We adjust the brightness in one of the two parts.  

Here is an example of shadowed image

<center>

![alt text][img_shadowed]

</center> 

This is implemented in `utils.random_shadow`.

##### Randomly adjust the image's brightness
We convert image into HSV color-space then adjust the V (brighness) value. The result image might look like

<center>

![alt text][img_brightness]

</center> 

This is implemented in `utils.random_brightness`.

##### Randomly translate the image
We translate the image in both x and y direction
* `dx` in x direction: we need to adjust the steering angle = 0.002 x dx
* `dy` in y direction: to simulate the road is up-hill/down-hill

Here is an example of translated image

<center>

![alt text][img_translated]

</center>

This is implemented in `utils.random_translate`.

##### Randomly select from left, center and right camera
From training data, we have left and right camera. As suggested from the project pages, one can use left/right
camera in addition to the center one to generate more data. The main idea is as the following

<center>

![alt text][multi_camera]

</center>

So we can compute steering for left/right camera image as

* `steering_left  = steering_center + correction`
* `steering_right = steering_center - correction`
 
where `correction` is a parameter to be tuned. We start with `correction = 0.2` as suggested in the project page.


#### Image preprocessing

##### Cropping
Looking at original image, we see the sky and the car itself which don't contain useful information for predicting the steering angle and it might add more noise. 
As suggested in the project page, we crop the top and the bottom of the image. After trying few times, we decide to cut the top 60 pixel and the bottom 25 pixel.

<center>

![alt text][img_cropped]

</center>

After cropped, the image will have shape (75, 320, 3).

##### Resizing
Note that from the [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/), they state the input shape is (66, 200). 
To following the same input as NVIDIA model, we will resize input image from (75, 320) to (66, 200), for example

<center>

![alt text][img_preprocessed]

</center>

By resizing the input from (75,320) to (66, 200) we reduce the parameters by half (from 559,419 to 252,219). This will helps with overfitting issue.

##### Normalizing
We will use a simple normalization to make input between range [-1,1] by define a Lambda layer x : x / 127.5 - 1.0, this is done in the first layer of the model.


### Model Architecture and Training Strategy

#### Model Architecture
As suggested in the project page, we start with [NVIDIA model](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) with the following architecure

```
Layer (type)                 Output Shape              Param #   
=================================================================
normalize (Lambda)           (None, 66, 200, 3)        0         
_________________________________________________________________
conv1 (Conv2D)               (None, 31, 98, 24)        1824      
_________________________________________________________________
conv2 (Conv2D)               (None, 14, 47, 36)        21636     
_________________________________________________________________
conv3 (Conv2D)               (None, 5, 22, 48)         43248     
_________________________________________________________________
conv4 (Conv2D)               (None, 3, 20, 64)         27712     
_________________________________________________________________
conv5 (Conv2D)               (None, 1, 18, 64)         36928     
_________________________________________________________________
flatten (Flatten)            (None, 1152)              0         
_________________________________________________________________
hidden1 (Dense)              (None, 100)               115300    
_________________________________________________________________
hidden2 (Dense)              (None, 50)                5050      
_________________________________________________________________
hidden3 (Dense)              (None, 10)                510       
_________________________________________________________________
output (Dense)               (None, 1)                 11        
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
```

The number of parameters is well above the number of samples, we might be overfitting. To combat it, we use data augmentation.
 
Later, we will test the model in Autonomous mode and tweak the model via trial and error. 

#### Training Strategy

In order to gauge how well the model was working, we split our dataset into a training and validation set. This helps to identify the problem of overfitting/underfitting. It also helps to choose model.

Here we don't use test set but we will test the model on the simulator.

In all training, we save down two trained models: one with best validation-loss and one after the last epoch. By default we call it `model_best.h5` and `model_last.h5`.

The default training parameters are

|               | default  |
|--------------:|:--------:|
| learning_rate |  1e-4    |
| batch_size    |  32      |
| epochs        |  20      |

We chose above default parameters since
 * `learning_rate=1e-4` since with the default `learning_rate=1e-3` training stops at high loss
 * `batch_size=32` seems working better than 64
 * `epochs=20` since we observe that validation-loss doesn't improve after that 

##### Experiment 1:
Using above model, we try the following augmentation setting

|                     | value     |
|--------------------:|:---------:|
| steering correction |  0.2      |
| dx range            |  [-40,40] |
| dy range            |  [-5,5]   |

We obtain



|               | loss     | val_loss |
|--------------:|:--------:|:--------:|
| model_best    |      |
| model_last    |  32      |


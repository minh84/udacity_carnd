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
* dx in x direction: we need to adjust the steering angle = 0.002 x dx
* dy in y direction: to simulate the road is up-hill/down-hill

Here is an example of translated image

<center>

![alt text][img_translated]

</center>

This is implemented in `utils.random_translate`.
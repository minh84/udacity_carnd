# **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[dataset_examples]: ./assets/dataset_examples.png

[color_hist_examples1]: ./assets/color_hist_examples1.png
[color_hist_examples2]: ./assets/color_hist_examples2.png
[color_hist_color_spaces]: ./assets/color_hist_color_spaces.png

[bin_spatial_rgb]: ./assets/bin_spatial_rgb.png
[bin_spatial_hsv]: ./assets/bin_spatial_hsv.png
[bin_spatial_hls]: ./assets/bin_spatial_hls.png
[bin_spatial_luv]: ./assets/bin_spatial_luv.png
[bin_spatial_yuv]: ./assets/bin_spatial_yuv.png
[bin_spatial_ycrcb]: ./assets/bin_spatial_ycrcb.png

[hog_rgb]: ./assets/hog_rgb.png
[hog_hsv]: ./assets/hog_hsv.png
[hog_hls]: ./assets/hog_hls.png
[hog_luv]: ./assets/hog_luv.png
[hog_yuv]: ./assets/hog_yuv.png
[hog_ycrcb]: ./assets/hog_ycrcb.png

[sliding_test_images]: ./assets/sliding_test_images.png
[sliding_windows]: ./assets/sliding_windows.png
[sliding_bbox1]: ./assets/sliding_bbox1.png
[sliding_bbox2]: ./assets/sliding_bbox2.png
[sliding_bbox3]: ./assets/sliding_bbox3.png
[sliding_bbox4]: ./assets/sliding_bbox4.png

[sliding_hls]: ./assets/sliding_bbox_hls.png
[sliding_hsv]: ./assets/sliding_bbox_hsv.png
[sliding_luv]: ./assets/sliding_bbox_luv.png
[sliding_yuv]: ./assets/sliding_bbox_yuv.png
[sliding_ycrcb]: ./assets/sliding_bbox_ycrcb.png

[heat_map_bbox]: ./assets/heat_map_bbox.png
[heat_map_filtering]: ./assets/heat_map_filtering.png

[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  

---
### Video Detection Pipeline
This project can be broken into 3 sub-tasks

* Build a classifier to detect an image is a car or not
* Run a sliding windows search to identify bounding boxes for vehicles in image
* Run above pipeline on a video


### Build a classifier
The first task is done in the notebook `vehicle_detection_p1_classifier.ipynb`, so here we only summary important points.

####1. Dataset
We use the labelled data given by Udacity: [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip). The dataset has following informations

| Name               | Value    |
| -------------:     |:--------:|
| number of cars     | 7709     |
| number of non-cars | 8968     |
| input shape        | (64,64,3)|
| input type         | float32  |
| input range        | (0,1)    |

The above info will be useful when we build our image-features.

Let's visualize some example of our dataset
<center>

![alt text][dataset_examples]

</center>

####2. Image features
As suggested in the project page, we could use the following features

* **color histogram**: which is implemented in `lesson_utils.get_color_hist`
* **binning spatial**: which is implemented in `lesson_utils.get_bin_spatial`
* **hog**: which is implemented in `lesson_utils.get_hog`

Note that, from previous projects, we know that `RGB` is subtle to lightning condition, so we should try other color-space such as HLS, HSV, LUV, YUV, YCrCb.

However each color space has **different range**, this is very important when we build **color histogram** or **hog**.

We choose to scale all image channel in any color-space to be between 0.0, 1.0 (`scaled=True`), this helps improving the classifier accuracy.

Let's visualize some image features

#####2.1 Color histogram
Here is an example of color-histogram 

<center>

![alt text][color_hist_examples1]
![alt text][color_hist_examples2]

</center>

It's not clear to distiguish car/noncar from the histogram, we also look at various color-space

<center>

![alt text][color_hist_color_spaces]

</center>

We use the following setting to build features

```python
hog_setting = {'orient'          : 9,
               'pix_per_cell'    : 8,
               'cell_per_block'  : 2,
               'vis'             : False,
               'feature_vec'     : True}

color_hist_setting = {'hist_bins'  : 32,
                      'bins_range' : (0., 1.)}

bin_spatial_setting = {'spatial_size' : (32, 32)}

combine_setting = {'hog'         : hog_setting,
                   'color_hist'  : color_hist_setting,
                   'bin_spatial' : bin_spatial_setting}
```

##### 2.2 Binning spatial
We explore the binning-spatial feature in various color-spaces

<center>

![alt text][bin_spatial_rgb]
![alt text][bin_spatial_hsv]
![alt text][bin_spatial_hls]
![alt text][bin_spatial_luv]
![alt text][bin_spatial_yuv]
![alt text][bin_spatial_ycrcb]

</center>

Look at above image, we see that depending on color-space, there is one channel that keeps the shape/constrast info better than these others, for example for HSV it's V-channel or for HLS it's L-channel. We will use this info when choosing the channel to compute HOG feature.

##### 2.3 HOG
We also look at HOG in different color spaces

<center>

![alt text][hog_rgb]
![alt text][hog_hsv]
![alt text][hog_hls]
![alt text][hog_luv]
![alt text][hog_yuv]
![alt text][hog_ycrcb]

</center>

The HOG capture some shape info of object in the image.

##### 2.4 Combining features
As suggested in the project page, we will combine hog, color-hist and bin-spatial to create image-features. This is implemented in the function `lesson_utils.extract_features`, note that we also augment the data by horizontal flip (using `np.flip`), this doubles the size of our samples.

#### 3 Train our classifier
We decide to use `svm.LinearSVC` from `sklearn` as our classifier, training with default parameters we obtain the following result
```
Training with HLS
--------------------------------------
Training input shape   (26683, 4932)
Validation input shape (6671, 4932)
Training time 9.66 seconds
Training accuracy    100.00%
Validation accuracy  99.12%
--------------------------------------

Training with HSV
--------------------------------------
Training input shape   (26683, 4932)
Validation input shape (6671, 4932)
Training time 10.99 seconds
Training accuracy    100.00%
Validation accuracy  99.19%
--------------------------------------

Training with LUV
--------------------------------------
Training input shape   (26683, 4932)
Validation input shape (6671, 4932)
Training time 19.10 seconds
Training accuracy    100.00%
Validation accuracy  98.71%
--------------------------------------

Training with YUV
--------------------------------------
Training input shape   (26683, 4932)
Validation input shape (6671, 4932)
Training time 16.50 seconds
Training accuracy    100.00%
Validation accuracy  98.95%
--------------------------------------

Training with YCrCb
--------------------------------------
Training input shape   (26683, 4932)
Validation input shape (6671, 4932)
Training time 19.34 seconds
Training accuracy    100.00%
Validation accuracy  98.86%
--------------------------------------
```
The classifier did reasonably well in all color space especially for `HLS` and `HSV` it got > 99% validation accuracy. However all of them seems suffering from overfitting.

We save our trained classifier so we can use it in the second task **Sliding-Window-Search**.

### Sliding Window Search
This task is done in `vehicle_detection_p2_sliding_window.ipynb` which covers the following

* define multi-scale search-windows
* run classifier on the search windows
* apply heat-map filtering to filter false-positive and multiple detection

#### 1. Multi-scale search windows

First, we take a look at the test images

<center>

![alt text][sliding_test_images]

</center>

We notice

* vehicle in image can appear to be small/big depending on its position
* vehicle is small if it's near horizontal line

Using above information, we will define the multi-scale windows that 

* small scale will be near horizontal line
* big scale will be close to bottom line

We try various sliding-search-windows scheme until it works on all test images. After a lot of trial/error we chose the following search windows

<center>

![alt text][sliding_windows]

</center>

The parameters for above sliding windows is
```python
scales        = [64, 96, 128, 160]
x_start_stops = [[412, None],   [None,   None], [None, None], [None, None]]
y_start_stops = [[400, 500]  ,  [400,  550],  [400,  600],  [400, 650]]
x_overlaps    =  [0.75, 0.75, 0.75, 0.75]
y_overlaps    =  [0.75, 0.5, 0.5, 0.5]
```

#### 2. Run classifier on search-windows
Given a set of search-windows, we search for windows that contain a car which is implemented in `lesson_utils.search_windows`. Here is the output
<center>

![alt text][sliding_bbox1]
![alt text][sliding_bbox2]
![alt text][sliding_bbox3]
![alt text][sliding_bbox4]

</center>

Looking at above we can see two issues: false-positive and multiple-detection. We will use heat-map filtering to solve these issues.

#### 3. Heat-map filtering
The heat-map filtering technique is implemented in `lesson_utils.heat_map_bbox` which uses the following observation: if 
there is car then bounding-boxes around the car has more probability to be detected than location without car. The function works well on the test images

<center>

![alt text][heat_map_bbox]

</center>

Note that we can integrate the heat-map for subsequent frame to make it more stable (which is what we do in the Video Implementation)

#### 4. Other color-spaces
In the first task `HSV` and `HLS` perform better than other color-space, however, they seem having more false positive than other method

<center>

![alt text][sliding_hls]
![alt text][sliding_hsv]
![alt text][sliding_luv]
![alt text][sliding_yuv]
![alt text][sliding_ycrcb]

</center>

So we chose `LUV` as the final color-space for Video Implementation.

### Video Implementation
We experiment the video-pipeline in `vehicle_detection_pipeline.ipynb` where we try various setting of `max_frames` and `threshold`.

We notice
* if `max_frames` is too high it might produce more false positive
* if `threshold` is too high/low it might produce more false negative/positive 

The final parameter is `max_frames=3, threshold=9`
####1. Final Video
The pipeline is implemented in `pipeline.pipeline_video_memory`, the final video is saved down to `project_video_out.mp4`. 

####2. Heat Map Filtering

I recorded the positions of positive detections in each frame of the video `pipeline.BoundingBoxes`. Then, from the 
positive detections from `max_frames` subsequent frames we created a heatmap and then thresholded (`threshold`) that map 
to identify vehicle positions. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

<center>

![alt text][heat_map_filtering]

</center>

---

##Conclusion

The pipeline works reasonably well, however we find the following limitation in our approach

* our classifier is overfitting and has bias to black car (there are more black car samples)
* sliding windows are fixed and might not optimal
* the pipeline is very slow (it took 1.3s/frame on my PC)
* the vehicle is only detected after it appears enough in the image (only a part of the car is not recognized)

Regarding speed, one could improve by computing the HOG feature to the whole region of interest then interpolate, due to 
time limit so we haven't tried this out. Also we could use Neural-Network for the classifier, we will experiment with this later when we have more time. 
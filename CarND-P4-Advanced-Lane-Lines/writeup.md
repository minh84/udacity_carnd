#**Advanced Lane Finding**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

[calib_cam]: ./assets/cam_calibrated.png "Calibrated camera raw v.s undistort"
[straight_line1]: ./test_images/straight_lines1.jpg "Straight lane line"
[step1_undistort]: ./assets/step1_undistort.png "Step 1"
[step2_bin_thresh]: ./assets/step2_bin_thresh.png "Step 2 Binary threshold"

[step3_perspec_trans]: ./assets/step3_perspective_trans.png "Step 3 Perspective transform"
[step3_bin_warped]: ./assets/step3_bin_warped.png "Step 3 Perspective transform of binary image"

[step4_lane_line_hist]: ./assets/step4_lane_line_hist.png "Step 4 Lane-line finding with histogram"
[step4_lane_line_conv]: ./assets/step4_lane_line_conv.png "Step 4 Lane-line finding with convolution"

[step6_draw_lane_line]: ./assets/step6_draw_lane_line.png "Step 6 Draw lane line"
[step6_test_pipeline]: ./assets/step6_test_pipeline.png "Step 6 Test pipeline"

[video_naive_failed]: ./assets/video_naive_failed.png "Failed frames 1"
[video_naive_improved1]: ./assets/video_naive_improved1.png "Improved Binary threshold"
[video_naive_failed2]: ./assets/video_naive_failed2.png "Failed frames 2"

[frame1045_bin]: ./assets/frame1045_bin.png "Frame 1045 binary"
[frame1045_bin_channel]: ./assets/frame1045_bin_channel.png "Frame 1045 binary in diffrent channel"
[frame1045_lane_finding]: ./assets/frame1045_lane_finding.png "Frame 1045 lane finding"

[frame587_bin]: ./assets/frame587_bin.png "Frame 587 binary"
## Rubric Points

Here I will consider the [rubric points]((https://review.udacity.com/#!/rubrics/571/view)) individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

To calibrate our camera, we need to use chessboard image since it has regular hight constrat pattern make it's easy
to detect automatically. Udacity kindly provides us sample images inside `camera_cal` folder. Here is the steps for 
camera calibration with OpenCv `cv2`
  1. load calibration images
  2. find inner-corners on the images (note that input chessboad has 9x5 inner-corners), denote these as image points
  3. construct object points which is fixed for all images
  4. use `cv2.calibrateCamera` to calibrate camera
  5. test calibrated coefficients

The camera calibration is implemented in `utils.calibrate_camera` and we can see how to use it in `advanced_lane_line.ipynb`. 

Now let's test the implementation 

![alt text][calib_cam]

The calibrated camera seems work well.

### Pipeline (single images)
The main task is to detect lane-line and car position using camera image. Its pipeline consists of the following sub-tasks
* undistort the raw image
* apply gradient threshold and color threshold technique to collect/filter useful information, this might 
use one/all of the techniques below (learnt from project pages)
    * absolute Sobel threshold in x/y direction
    * magnitute Sobel threshold
    * direction threshold
    * color threshold using S-channel of HLS color-space
* find the lane-line
* measure curvature-radius and car's position with respect to lane center

As learnt from the project-page, we will use a straight-lane-line image as input since 
when we apply perspective transform we know that output lane-line will be two straight 
and parallel lines. Here is the raw image input

![alt text][straight_line1]

Once we are happy with the pipeline, we will test the pipeline on all test images (in sub-folder `test_images`)

#### 1. Example raw image and undistort one

Here is the distortion-corrected image of the above input

![alt text][step1_undistort]

This is hard to see the difference between two images by eyes. 

#### 2. Threshold binary images

In the first project, to find lane-line we use Canny edges which works reasonably well 
in normal condition (no shadow, white lane-line, road is straight e.t.c). Now we will look at more advanced 
techniques so that our lane-line detector

 * can work in any lighting condition with/without shadow
 * can work for curvy road
 * can work for different lane-line (yellow/white)
 
The technique we use to create a binary image is a combination of 

 * absolute Sobel threshold in x/y direction
 * magnitute Sobel threshold
 * direction threshold
 * color threshold using S-channel of HLS color-space

The course has explained well the above technique so we don't repeat it here. The implementation is done in `utils.binary_threshold_v1`.

Here is the output of above undistorted image 

![alt text][step2_bin_thresh]

#### 3. Perspective transform

To better identify the lane line, it will be easier to view the road from bird-eye view since
* in camera view, lane lines seem crossing at the farther end
* in bird-eye view, lane lines will be parallel lines and seperated in both end 

The code for my perspective transform is implemented in `utils.warper` using the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 205, 720      | 300, 720      | 
| 595, 450      | 300, 0        |
| 685, 450      | 950, 0        |
| 1110,720      | 950, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][step3_perspec_trans]

And here is the perspective transform of the binary image

![alt text][step3_bin_warped]

#### 4. Identify lane-line
Using the above binary-warped image, we will apply one of the following technique to find lane-line

* historgram of non-zero pixel
* sliding convolution of hot-pixel

Both of the two techniques are well explained in the course. We implement them in `find_lane_line_hist` and `find_lane_line_conv` in `lane_lines.py`.

The fitted lane-lines looks like this


![alt text][step4_lane_line_hist]
![alt text][step4_lane_line_conv]

In this example, both methods works well. 

#### 5. Measuring curvature and car postion

The radius of curvature is well explained in [this tutorial](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). We implement it in `measure_curvature`. Note here we transform from pixel to meter using the following constant (taken from the project page)

| Direction | Pixel to Meter | 
|:---------:|:--------------:| 
| x         | 3.7/700        | 
| y         | 30.0/720       |

To measure the car position, we can assume that the camera is mounted at the center of the car, such that the lane center 
is the midpoint at the bottom of the image between the two lines you've detected. 

The offset of the lane center from the center of the image (converted from pixels to meters) is your distance from the center of the lane. The implementation is done in `measure_offset`.

we obtain the following result for above image

```
Left-right radius(m)   = 5408, 2411
Car position offset(m) = 0.07 right of the center
```

#### 6. Draw it onto the road

Now we know the lane lines on the binary-warped image, the curvature's radius and the car position. We can draw it back to road image.
We implement this step in function `draw_lane_area` in `lane_lines.py`.

Here is the output for above image:


![alt text][step6_draw_lane_line]

We put all the above steps into `pipeline_single_image` in `pipeline.py`. The lane-line for all test images looks like

![alt text][step6_test_pipeline]

The pipeline works reasonably well on all test images. Now let's try it with `project_video.mp4`. 

---

### Pipeline (video)

#### 1. Naive approach
We can try the `pipeline_single_image` on each frame of the clip, this is implemented in `pipeline_video_naive` in `pipeline.py`.

The output is saved to `project_video_naive_v1.mp4`. This approach works reasonably well except around the 42-nd second (or frame 1042 - 1051), it fails to detect lane-line as seen below

![alt text][video_naive_failed]

Let's look at frame-1045th

![alt text][frame1045_bin]

![alt text][frame1045_lane_finding]

It's clear that the shadow make the lane-line finding failed to find the right lane-line.

#### 2. Review binary threshold

We review the color threshold of the frame 1045th and we find that

![alt text][frame1045_bin_channel]

It seems that if we combine both H and S color, it might improve the lane-line detector. And indeed, the result looks better

![alt_text][video_naive_improved1]

Looking at the video output `project_video_naive_v2.mp4`, we find that at the 24-th second (or frame 585 - 595), it fails to detect the left-lane

![alt_text][video_naive_failed2]

The binary threshold of the frame 587th looks like

![alt text][frame587_bin]

It's clear that due to the constrast of the left wall, it makes the wall to appear in the binary image which causes the lane-line detection failed.

There are two ways to work around the above issue

 * defining a fixed region of interest, this might work well on this track since we drive well in middle of the lane and the road is almost straight.
 * noticing that in the above frames, it fails to detect lane-line in only 3 frames (587,588,589) but in other frames it works well. We can then use previous lane line as starter point to search for lane-line in current frame. This serves as dynamic region of interest which might work well for sharp curves and tricky condition.
  
We try to implement the second approach now.
  
#### 3. Look ahead-filter
As suggested in the project page, we should keep track of the last few lane-line detection and do the following sanity check

* checking that they have similar curvature
* checking that they are separated by approximately the right distance horizontally
* checking that they are roughly parallel

Also, we can use the lane-line of the previous frame as a start point for the next frame, this will ensure a smooth transition of 
lane-line detection and also help to filter out noisy result.

The implementation is done in
* class `lane_line.LaneLine` to allow us to keep track of previous fitted parameters
* function `pipeline.pipeline_img_with_memory` which detects lane-line for a frame using fitted information from previous frame
* function `pipeline.pipeline_video_look_ahead` which detects lane-line for each frame in a video

The result is saved to `project_video_look_ahead.mp4`

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

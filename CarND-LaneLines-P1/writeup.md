# **Finding Lane Lines on the Road** 

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on current approach

[imageIn]: ./test_images/solidWhiteCurve.jpg "Input"
[imageOut]: ./test_images_output/solidWhiteCurve.jpg "Output"

[image2]: ./examples/grayscale.jpg "Grayscale"

[imgChallengFrame0]: ./report_images/challenge_frame_0.png "Challenge frame 0"

[roi1]: ./report_images/region_oi_1.png "Region 01"
[roi2]: ./report_images/region_oi_2.png "Region 02"

[out1]: ./test_images_output/solidWhiteCurve.jpg "out 1"
[out2]: ./test_images_output/solidWhiteRight.jpg "out 2"
[out3]: ./test_images_output/solidYellowCurve.jpg "out 3"
[out4]: ./test_images_output/solidYellowCurve2.jpg "out 4"
[out5]: ./test_images_output/solidYellowLeft.jpg "out 5"
[out6]: ./test_images_output/whiteCarLaneSwitch.jpg "out 6"

## Problem description

We start by finding lane lines on the road from a static image

![alt text][imageIn]

The goal is to detect the lane lines that we are currently in and mark it by some color (red). 
For example, we want to obtain the following output 

![alt text][imageOut]

Then later, we can apply the lane-finding to each frame of a video.

## Reflection

### Lane-finding pipeline

As we have learnt from lectures (Udacity), the pipeline consists of 6 steps
1. First, we convert input-image to grayscale
2. Then, we apply Gaussian-smoothing, note that we have a parameter that need to be tuned (the smoothing kernel-size). We use 3 as learnt from lectures.
3. Next, we can use `canny-edge-dection` technique, we again use the same parameters from lecture i.e `low_threshold=50, high_threshold=150`
4. Now, we need to define two regions of interest one for the left-lane and the other for the right-lane
5. One each region, we will use `cv2.HoughLinesP` on the edge image to detect lines. Here, after few tunning and go through forum's suggestion, we use the following parameters
    * rho = 1
    * theta = pi/180
    * threshold = 1 
    * min_line_len = 5
    * max_line_gap = 10
   
   We notice that the following behaviour:
    * if we increase threshold, the mark line becomes less stable
    * if we increase min_line_len, we might have no line at all
       
6. Finally, using the lines from previous steps, we need to average/extrapolate. This part is done in `draw_solid_lines`

In above pipeline, there are two parts that different from lectures (4-th) and (6-th) steps. So we will focus more on these two steps.

#### Region of interest
We first try the quadrilateral with following vertices
* x_bottom_left = 0.05 * x_size
* x_bottom_right = 0.95 * x_size
* x_top_left = 0.48 * x_size
* x_top_right = 0.52 * x_size
* y_bottom = y_size
* y_top = 0.6 * y_size

The mask of region of interest has following form

![alt text][roi1]


We find that it works well on test images and the two videos `solidYellowLeft.mp4` and `solidWhiteRight.mp4`.

However, for the challenge video, it doesn't work at all

![alt text][imgChallengFrame0]

The reason is due to in the challenge video, the bottom of each image contains also a part of car which introduce **noise** into the region of interest.
 
One quick work around is to reduce the region of interest. Plus, we think the lane will make a V-shape, so we consider the polygon defined in three functions `get_yrange, get_xrange_left, get_xrange_right`. 
The region of interest becomes

![alt text][roi2]

With new roi2, it's still working well on two previous video and also it works a bit better on the challenge video.

#### Draw solid lines
We want to average/extrapolate the lines obtained from `cv2.HoughLinesP`. Since we expect these points of these lines are belong to a line, so to extrapolate, one can use following steps
 * we can do a linear fitting on these points
 * using fitted parameters we can extend to the limit of x, y. In the code, we use y_max, y_min (this they are shared between left and right lane-lines) and compute x_max, x_min given fitted parameters.
 
When doing linear fitting, we observe that the input can contain outliner so we use a simple outline-removal algorithm based on variance (filter out point that has error not in [m-2*sig, m + 2*sig] where m, sig are mean and stddev or the fitter errors).
This makes the fitted line be more robust.
 
#### Test on test images
Before implement on videos, we test it out on images

![alt text][out1]
![alt text][out2]
![alt text][out3]
![alt text][out4]
![alt text][out5]
![alt text][out6]

### Shortcomings
When applying the pipeline on the challenge video, we still see that the two lane-lines are not stables. This happens when we went through the road that has tree's shadow. 
The tree shadow introduces more noisy edge-lines which is causing lines' slopes is jumping around.

Another shortcomming is the way we define region of interest. We hard corded some parameters that might not work well with other setting of the camera (position, resolution). 

### Future steps

We think that we can improve the pipeline by doing the following steps
* since we know that the lane line are either white or yellow, one could employ some technique that filter only these two color then we can detect line & edge more easily. 
* we could use a segmentation algorithm to segment which is the part of the road, then define the region of interest only on this road-region.
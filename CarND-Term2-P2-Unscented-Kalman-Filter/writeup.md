# **Unscented Kalman Filter**

The goals / steps of this project are the following

* Implement [Unscented Kalman Filter](https://en.wikipedia.org/wiki/Kalman_filter#Unscented_Kalman_filter) to estimate state of a CTRV model (constant turn-rate and velocity magnitude model)
* We apply Unscented Kalman Filter using signal from Laser and Radar sensor
* We test/run it using [Udacity's SDC simulator](https://github.com/udacity/self-driving-car-sim/releases/)

[lidar_nis1]: ./assets/lidar_nis1.png
[lidar_nis2]: ./assets/lidar_nis2.png
[radar_nis1]: ./assets/radar_nis1.png
[radar_nis2]: ./assets/radar_nis2.png

## Rubric Points
Here we will consider the [rubric points](https://review.udacity.com/#!/rubrics/783/view) individually and describe how we addressed each point in my implementation.

  
## Submission Files
The EKF is implemented in the following files

* **ukf.h/cpp**: implemented UKF on Lidar/Radar signal 
* **tools.cpp**: an utility to compute RMSE

## Parameters
For process parameters, we use the rule of thumb (as suggested in lesson) to set acceleration noise and deviation yaw acceleration to 
```
// Process noise standard deviation longitudinal acceleration in m/s^2
std_a_     = 3.;

// Process noise standard deviation yaw acceleration in rad/s^2
std_yawdd_ = M_PI/4.;
```

We computed NIS and write it to stdout, we can collect it via the following command
```
./UnscentedKF | awk '/nis/{print}' > ../run.csv
```

We write a small jupyter-notebook (`nis_analysis.ipynb`) to visualize Lidar/Radar's NIS profile.

For Lidar, We keep the same parameter as starter code, since we visualize its NIS and it looks sane

<center>

![alt text][lidar_nis1]

</center>

For Radar, with starter parameter, we obtained the following NIS profile

<center>

![alt text][radar_nis1]

</center>

It seems that we under-estimate the radar-noise a bit, so we increase radar-noise 
```
// Radar measurement noise standard deviation radius in m
std_radr_ = 0.35;

// Radar measurement noise standard deviation angle in rad
std_radphi_ = 0.035;

// Radar measurement noise standard deviation radius change in m/s
std_radrd_ = 0.35;
```

The Radar's NIS becomes

<center>

![alt text][radar_nis2]

</center>
which looks more reasonable.

So the final chosen parameters are

| Parameters | Value |
|-----------:|:------|
| std_a      | 3.    |
| std_yawdd  | pi/4  |
| std_laspx  | 0.15  |
| std_laspy  | 0.15  |
| std_radr   | 0.35  |
| std_radphi | 0.035 |
| std_radrd  | 0.35  |

And we obtain the following RMSE

| State | RMSE  |
|------:|:-----:|
| X     | 0.0752|
| Y     | 0.0852|
| VX    | 0.3259|
| VY    | 0.2408|

## Conclusion
We can see UKF gives better RMSE than EKF (see last project) which is expected since UKF captures the mean/variance of transformed GRV (passed through a non-linear function) better than EKF.
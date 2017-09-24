# **Extended Kalman Filter**

The goals / steps of this project are the following

* Implement [Extended Kalman Filter](https://en.wikipedia.org/wiki/Extended_Kalman_filter) to estimate position & velocity of object in 2D space
* We apply Extended Kalman Filter using signal from Laser and Radar sensor
* We test/run it using [Udacity's SDC simulator](https://github.com/udacity/self-driving-car-sim/releases/)

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/748/view) individually and describe how I addressed each point in my implementation.
  
## Submission Files
The EKF is implemented in the following files

* **FusionEKF.cpp**: processing Lidar/Radar measurement to predict/update the state (i.e position/velocity) of our car
* **kalman_filter.cpp**: predict/update steps for the (Extended) Kalman Filter
* **tools.cpp**: some  utility to compute Jacobian & RMSE
* **measure_model.h/cpp**: represents Laser/Radar measurements

The code is self-explained and very similar to lecture code-quiz, however we need to handle the following points

* The **Radar**'s measurement is (rho, phi, rhodot) and our state is (px, py, vx, vy), so we need to convert *cartesian to polar* before computing the error. 
* Also since phi is in radian unit, we need to **normalize it in range [-pi, pi]**, this is one crucial step to make Fusion Lidar/Radar working.
* The *main.cpp* doesn't handle well the client's disconnection (Segmentation Fault), it seems due to the line **ws.close();**, so we comment it out and it's now working fine (it can handle disconnection well).
* Note that we need to initialize the state/covariance at time 0, and we use the first measurement to initialize the state . However the covariance can be anything, so we re-use input from quiz-code

```python
P0 = (1, 0, 0, 0,
      0, 1, 0, 0,
      0, 0, 100, 0,
      0, 0, 0, 100)
```
* For **Radar** we need to compute the Jacobian **Hj = dRadar/dState**, however Hj is not defined when rho=0. At the moment set Hj = previous Hj (see *RadarMeasure::jacobian*).

We also improved the code to support the following points
* The simulator allows to **restart a run** but our simulator doesn't receive an explicit message saying "simulator restarted", so we verify it by checking if **current-time < previous-time** (see *FusionEKF::ProcessMeasurement* for more detail).
* The simulator has two datasets, but the server doesn't know how to reset when switching datasets, so you need to **restart server** before switching dataset.

We restructure code base a bit by introducing **RadarMeasure/LaserMeasure**
* we can simplify **KalmanFilter**: function *UpdateEKF* can be removed.
* we can simplify **FusionEKF**: removed all matrix and simplify *FusionEKF::ProcessMeasurement*, this can allow to fuse more measurements.

## Results
Using EKF, we obtain the following RMSE

| Variable | RMSE-Dataset1 | RMSE-Dataset2 |
| --------:|:-------------:|:-------------:|
| X        | 0.0977        | 0.0729        |
| Y        | 0.0854        | 0.0963        |
| VX       | 0.4406        | 0.3875        |
| VY       | 0.4608        | 0.4677        |

## Experementing with IEKF
We try to improve EKF with Iterative Extended Kalman Filter from this [paper](http://iopscience.iop.org/article/10.1088/1742-6596/659/1/012022/pdf). The RMSE is improved a bit

| Variable | RMSE-Dataset1 | RMSE-Dataset2 |
| --------:|:-------------:|:-------------:|
| X        | 0.0944        | 0.0729        |
| Y        | 0.0847        | 0.0956        |
| VX       | 0.3847        | 0.3863        |
| VY       | 0.4049        | 0.4381        |

Here we use 5 iteration in each update step for Radar measure.

## Conclusion
After completing the implementation EKF, we can see the EKF is straightforward but very powerfull in estimating current state. We also learn how to fuse multiple measurement (Laser/Radar) in order to track object's position/speed.
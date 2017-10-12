#**PID Control**
The goals / steps of this project are the following:
* Implement PID control to drive the car in simulator given CTE (cross-track-error)

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/824/view) individually and describe how I addressed each point in my implementation.

## Submission Files
The PID is implemented in `pid.h/.cpp` where we compute the control using [PID](https://en.wikipedia.org/wiki/PID_controller).

The main functions are  

* `PID::Init`: to setup `Kp, Ki, Kd` parameters
* `PID::UpdateError`: to update the error-term `p_error`, integral of errors `i_error` and differential of errors `d_error`.
* `PID::TotalError`: to compute the control input which has the following form
<center>
u = - (Kp * p_error + Ki * i_error + Kd * d_error)
</center>

The PID-control is used to compute the `steering angle` and the `throttle` to make the car drive at an desired speed and in the centre of the road.

## PID-Observation
We observe that
* Kp is too small: the car might not be able to drive through shape turn
* Kp is too large: the car might not drive smoothly (keep turning left/right)
* Kd does help on sharp turn, but too large Kd also cause bumpy drive (keep turning left/right)
* Ki need to be small: we observe that `i_error` is mostly positive
* Kp, Kd, Ki depend on the desired speed

Using above observation to manually tune parameters (similar as Twiddle but manually done). We fix desired speed to be **20MPH**, and tune `Kp, Ki, Kd` to make the car drive as smooth as posssible.

The final parameter we use is 
```
Kp = 0.15
Ki = 0.002
Kd = 0.8
```

To obtain the desired speed, we use the PID from Behaviroral-Cloning i.e `Kp=0.1,Ki=0.002,Kd=0`.
 
We find that above PID can be used with speed up to **30MPH** but we can see the drive is less smooth than with **20MPH**.

## Conclusion
PID is very easy to implement and powerful enough to control our car to desired level (speed/position). 

#**Model Predictive Control**

The goals / steps of this project are the following

* Implemenet the kinematic model to represent the dynamics of the vehicle's state. The kinematic model ignores tire forces, gravity and mass which reduces the acuracy of the models, but it also makes them more tractable. 
* Using the kinematic model, we frame the task of following a trajectory as an optimization problem. The solution to the optimization problem gives us an approximately optimized control.
* We test/run it using [Udacity's SDC simulator](https://github.com/udacity/self-driving-car-sim/releases/).

[run1]: ./assets/run1.png
[run1_step25]: ./assets/run1_step25.png
[run1_step26]: ./assets/run1_step26.png
[run1_step27]: ./assets/run1_step27.png
[run1_step28]: ./assets/run1_step28.png

[run1_step27_delta]: ./assets/run1_step27_delta.png
[run1_step27_acc]:   ./assets/run1_step27_acc.png

[run1_verrs]: ./assets/run1_verrs.png
[run1_cte_epsi]: ./assets/run1_cte_epsi.png
[run1_acc_delta]: ./assets/run1_acc_delta.png
[run1_acc_delta_diff]: ./assets/run1_acc_delta_dif.png

[run2]: ./assets/run2.png

[run3]: ./assets/run3.png
[run4a]: ./assets/run4a.png
[run4b]: ./assets/run4b.png

## Rubric Points
Here we will consider the [rubric points](https://review.udacity.com/#!/rubrics/896/view) individually and describe how we addressed each point in my implementation.

### The Vehicle Model
In this project we use the kinematic model to describe the vehicle-state as following
 * The vehicle-state is represented by a vector `(x,y,psi,v,cte,epsi)` where
    * `x,y` is the vehicle's position
    * `psi` is the heading angle
    * `v` is the velocity
    * `cte` is the cross-track-error
    * `epsi` is the heading-error
    
 * The vehicle-state evoles through time from `t` to `t+dt` as following
    ```
    x(t+dt)   = x(t) + v(t) * cos(psi(t)) * dt
    y(t+dt)   = y(t) + v(t) * sin(psi(t)) * dt
    psi(t+dt) = psi(t) + v(t)/Lf * delta(t) * dt
    v(t+dt)   = v(t) + a(t) * dt
    ```
    where `delta, a` is turn-rate and accelerator respectively. The errors' dynamics is given as following
    ```
    cte(t+dt)  = f(x(t)) - y(t) + v(t) * sin(epsi(t)) * dt
    epsi(t+dt) = psi(t) - psides(t) + v(t)/Lf * delta(t) * dt
    ```
    looking at the above errors' dynamics we notice that
    * `f(x(t)) - y(t)` and `psi(t) - psides(t)` being the current error
    * `v(t) * sin(epsi(t)) * dt` and `v(t)/Lf * delta(t) * dt`being the change in error caused by the vehicle's movement 

    Notice that in the above equations, we need to define `f(x(t))`-the reference trajectory. However, in this project, we are given a list of way-points (received in the Simulator's message) as the points to be followed. So to define `f(.)`, one can fit a polynomial to these way-points.
       
So we have described the vehicle model's state and dynamics, the implementation is listed in the below
 * the vehicle model's state-dynamics is implemented in `MPC.cpp (l. 141-144)`.
 * the vehicle model's error-dynamics is implemented in `MPC.cpp (l. 147-148)`.
 * the polynomial fitting is implemented in `main.cpp (l. 114)` and the evaluation of polynomial (and its derivation) is implemented in `poly_utils.cpp`.
  
### MPC as Optimized Problem
In the MPC lecture, we frame the task of following way-points as an optimized problem: finding `delta(t), a(t)` that minimized a cost function `J(t)`. The main task is then becoming how to define `J(t)`, one can consider the following

* **Cross Track Error**:
    ```
    cte(t)^2 + cte(t+dt)^2 + ... + cte(t+(N-1)dt)^2
    ```
    to track the difference between current position and reference trajectory.
* **Heading Error**:
    ```
    epsi(t)^2 + epsi(t+dt)^2 +  ... + epsi(t+(N-1)dt)^2
    ```    
    to track the difference between current heading and reference heading.
* **Reference Velocity Error**:
    ```
    (ref_v - v(t))^2 + ... + (ref_v - v(t + (N-1)dt))^2
    ```
    
* **Regularization**
    ```
    delta(t)^2 + ... + delta(t + (N-2)dt)^2
    a(t)^2 + ... + a(t+(N-2)dt)^2    
    ```
    to minimize the actuators input.
    ```
    (delta(t+dt) - delta(t))^2 + ... + (delta(t + (N-2)dt) - delta(t + (N-3)dt))^2
    (a(t+dt) - a(t))^2 + ... + (a(t + (N-2)dt) - a(t + (N-3)dt))^2
    ```
    to enhance the smoothness of the actuators.

One can define `J(t)` as a combination of all above (with different associated weights. There are two parameters to be considered here `dt` and `N` which will be tuned depends on the vehicle's velocity.

Note that, to model a real life situation, we introduce a latency of 100ms. How we modify the optimization problem to handle this latency. It's actually very straightforward, for example consider `dt=50ms` then `a(t), delta(t)` reached the system at time `t+2*dt`, and `a(t), a(t+dt), delta(t), delta(t+dt)` are fixed by previous step.
  
The implementation is listed below
* cost function `J(t)` is implemented in `MPC.cpp (l. 68-80)`
* latency-handle is implemented in `MPC.cpp (l. 16,260-268)`

### Simulator Testing/Parameters Tuning
As described above, we have the following tunning parameters
* `dt` and `N`
* associated weights with different errors

We start by using `dt=0.05, N=25` and all associated weights `= 1.0` and reference-speed `ref_v=30`, we observe that the car change the direction too much after just a few seconds

<center>

![alt text][run1]

</center>

We dump out the telemetry and the Iopt-solution to investigate further. We find that (see `mpc_visualize.ipynb` for more detail) 
* From step 25-th, the optimized trajectory becomes un-smooth

<center>

![alt text][run1_step25]
![alt text][run1_step26]
![alt text][run1_step27]
![alt text][run1_step28]

</center>

* Looking at the acctuators at step 27-th

<center>

![alt text][run1_step27_delta]
![alt text][run1_step27_acc]

</center>
We find that turn-rate (delta) is flipping from maximum to minimum, we need to penalize this behaviour to make our car to drive more smoothly.

* We visualize the error contribution

<center>

![alt text][run1_verrs]
![alt text][run1_cte_epsi]
![alt text][run1_acc_delta]
![alt text][run1_acc_delta_diff]
</center>

we can see that for the first few steps, the error for velocity is the main contribution which explains that the Iopt-solver failed to find optimal solution for minimizing cte.

From above analysis, it's clear that we need to penalize the abrupt change in turn-rate. This can be done by applying a higher weight to delta-difference term
 ```
 delta_w * [(delta(t+dt) - delta(t))^2 + ... + (delta(t + (N-2)dt) - delta(t + (N-3)dt))^2] 
 ```
We try `delta_w=100` and we can see that the car can drive smoothly through the lap
<center>

![alt text][run2]
</center>
However, the velocity 30MPH seems very slow, can we drive faster.

#### Tuning for v=50MPH
We change `ref_v=50` and we find that `dt=0.05` and `N=25` are too large. We optimize the trajectory to a point to far in the furture which might be outside of the road as shown in the image below
<center>

![alt text][run3]
</center>

we reduce `N` until the blue line ensure is less than the yellow line, we find that with `N=15`, the car can drive smoothly through the lap.

#### Tuning for v=80MPH
We change `ref_v=80`  and we also change `N=10`, `delta_w=500` the car manages to drive through the lap but it drives very close to the curve when there is a sharp turn as shown in below image
<center>

![alt text][run4a]
</center>
In the consequence, to fix itself, it drives to the curves in the next scene
<center>

![alt text][run4b]
</center>

In real life, when there is a sharp turn or we are in the wrong position (`cte` is big), we should slow our vehicle down. We implement this in `MPC.cpp (l. 295-296)` where we will set reference speed to be 50 when `|cte| > 1`.
With this dynamic reference speed, the car behaves reasonably well.

Note that UK national limit is 70MPH so we stop at 80MPH.

### Command line parse to MPC
To make it easier to change speed and turn on/off debug message, we use `cvxopt.hpp` from [here](https://github.com/jarro2783/cxxopts), this allows us to
 * set the reference speed with flag `-s or --speed   (default = 50)`
 * set the lower speed with flag `-l or --lower_speed (default = 40)`
 * turn on/off verbose with flat `-v or --verbose     (default = False)`
For example
```
        ./mpc -s 80 -l 50
```
will set `reference_speed=80 and lower-speed=50`.

We tested the code with `reference speed = 50, 60, 70, 80` and the car can drive through the lap without going over the curve.

## Conclusion
Doing MPC is a great experience, we learnt that we should use difference reference speed depends on the car's position (which makes sense). We also learnt how to tweak the cost function to ensure smooth solution. We handle the latency in a very simple way but it works reasonably well. We think the MPC can be improved further by using more realistic dynamics model and better cost function. 
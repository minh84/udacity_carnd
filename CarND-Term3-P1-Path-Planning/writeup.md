# **Path Planning**

The goals / steps of this project are the following
* Implement a path-planing that safely navigates our car in the simulation

This write up summarizes main points of this project, for more detail please check `research/path_planing.ipynb`

## Rubric Points
Here we will consider the [rubric points](https://review.udacity.com/#!/rubrics/1020/view) individually and describe how we addressed each point in my implementation.

## How we implement Path Planning
The path planning can be broken into two parts
* **trajectory generation**: to generate smooth trajectory that ensures limit on speed/acceleration and jerk.
* **behavior planner**: to consider surrounding car so that we ensure to drive safely and efficiently

### Trajectory generation
The trajectory generation is implemented in `trajectory.h/.cpp`, after various experiments, we choose `spline` method to generate a smooth trajectory.

The Spline allow us to generate smooth trajectory but we can't control speed exactly. That's why we set
* speed limit to 49.80MPH so the speed is always < 50MPH
* max acceleration is 5m/s2

### Behavior planner
This is the most tricky part, to ensure driving safely, we need to consider the following case
* a car, is in our lane and ahead of us, that might block us
* a car can change to our lane that might block us
* a car can change to the lane, which we want to change to, that might block us

We also need to consider when to change lane, we only change lane if
* that lane gives better speed
* no car is too close to us in that lane
* we faster than the car behind in that lane and slower than the car ahead in that lane.

The implementation is done in `behavior.h/.cpp`.

# Conclusion
This project is the most challenging project in the whole terms. It's challenging due to
 * there is a lot of freedom to implement trajectory generation and behavior planner
 * there is a lot of cases to be considered to ensure safely driving

We also learn a lot from this interesting project on how to use Spline to generate data and how to implement a static behavior planner. The car can drive safely (in our limited testing), but it could be improved futher by looking in longer range or by tweaking safety parameters.
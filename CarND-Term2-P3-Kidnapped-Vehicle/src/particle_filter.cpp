/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>
#include <unordered_set>
#include "particle_filter.h"

using namespace std;
static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double stddev[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	num_particles = 1000;
	weights = vector<double>(num_particles, 1.0);
	
	std::normal_distribution<double> normx(0., stddev[0]);
	std::normal_distribution<double> normy(0., stddev[1]);
	std::normal_distribution<double> normtheta(0., stddev[2]);

	for(int i = 0; i < num_particles; ++i) {
		Particle p;
		p.id    = i;
		p.x  	  = x + normx(gen);
		p.y     = y + normy(gen);
		p.theta = theta + normtheta(gen);
		p.weight = 1.0;

		particles.push_back(p);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	const double EPS=1.0e-6; // handle the case yawrate ~ 0.0
	
	std::normal_distribution<double> normx(0., std_pos[0]);
	std::normal_distribution<double> normy(0., std_pos[1]);
	std::normal_distribution<double> normtheta(0., std_pos[2]);

	if (fabs(yaw_rate) > EPS) {
		for (int i = 0; i < particles.size(); ++i) {
			double theta     = particles[i].theta;
			double new_theta = theta + yaw_rate * delta_t;
			double voveryawd = velocity / yaw_rate;

			// predict state + noise
			particles[i].x    += voveryawd * (sin(new_theta) - sin(theta)) + normx(gen);
			particles[i].y    += voveryawd * (cos(theta) - cos(new_theta)) + normy(gen);
			particles[i].theta = new_theta + normtheta(gen);
		}
	} else {
		for (int i = 0; i < particles.size(); ++i) {
			particles[i].x += velocity * cos(particles[i].theta) + normx(gen);
			particles[i].y += velocity * sin(particles[i].theta) + normy(gen);
			particles[i].theta += normtheta(gen);
		}
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

}

std::vector<LandmarkObs> carToMapCoorinate(const std::vector<LandmarkObs>& observations,
		 																			 const Particle& p) {
	// reserved enough space
	std::vector<LandmarkObs> obs_map_coor;
	obs_map_coor.reserve(observations.size());
	
	double costheta = cos(p.theta);
	double sintheta = sin(p.theta);

	for(auto obs : observations) {		
		LandmarkObs obs_m = obs;
		//convert from local car-coordinate -> map-coordinate
		obs_m.x = p.x + costheta * obs.x - sintheta * obs.y;
		obs_m.y = p.y + sintheta * obs.x + costheta * obs.y;

		obs_map_coor.push_back(obs_m);
	}

	return obs_map_coor;
}

std::vector<LandmarkObs> landmarkInRange(double sensor_range,
																				 const Particle& p,
																				 const Map& map_landmarks) {
	std::vector<LandmarkObs> retval;
	for (auto landmark : map_landmarks.landmark_list) {
		if (dist(p.x, p.y, landmark.x_f, landmark.y_f) < sensor_range) {
			LandmarkObs lm_in_range;
			lm_in_range.x  = landmark.x_f;
			lm_in_range.y  = landmark.y_f;
			lm_in_range.id = landmark.id_i;

			retval.push_back(lm_in_range);
		}
	}
	return retval;
}

std::vector<std::pair<double, double>> distanceToNearestNeighbor(const std::vector<LandmarkObs>& predicted,
																																 const std::vector<LandmarkObs>& observations) {
	std::vector<std::pair<double, double>> retval;
	retval.reserve(observations.size());
	for(auto obs : observations) {
		double best_dist = 1e10;
		double x = 0., y = 0.;
		for(auto lm : predicted) {
			double lm_dist = dist(obs.x, obs.y, lm.x, lm.y);
			if(lm_dist < best_dist) {
				best_dist = lm_dist;
				x = lm.x;
				y = lm.y;
			}
		}

		retval.emplace_back(x - obs.x, y - obs.y);
	}
	return retval;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	int nb_particles = particles.size();
	double cst = 1.0/(2.0 * M_PI * std_landmark[0] * std_landmark[1]);
	for (int i = 0; i < nb_particles; ++i) {		
		// get landmark in range
		std::vector<LandmarkObs> predicted = landmarkInRange(sensor_range, particles[i], map_landmarks);
		// convert observed landmark from local car-coordinate to map-coordinate
		std::vector<LandmarkObs> obs_in_map_coor  = carToMapCoorinate(observations, particles[i]);
		
		// compute distance to nearest neighbor since we only need distance to compute Multi-Gaussian probability
		std::vector<std::pair<double, double>> dist_to_nn = distanceToNearestNeighbor(predicted, obs_in_map_coor);

		// compute weight
		double prob = 1;	
		for (auto dist : dist_to_nn) {
			double exponent = dist.first*dist.first  / (2.0*std_landmark[0]*std_landmark[0]) +
													dist.second*dist.second / (2.0*std_landmark[1]*std_landmark[1]) ;
			prob *= (cst * exp(-exponent));
		}		
		
		// update weight
		particles[i].weight = prob;
		weights[i] = prob;
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	std::discrete_distribution<int> d(weights.begin(), weights.end());

	vector<Particle> new_particles;

	for(int i = 0; i < particles.size(); ++i) {
		int j = d(gen);
		new_particles.push_back(particles[j]);
	}

	// update resampled particles & weights
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

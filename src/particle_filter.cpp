/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	const int NUM_PARTICLES = 100;
	default_random_engine gen;

	num_particles = NUM_PARTICLES;
	particles.resize(num_particles);

	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);	
	for (int i = 0; i < num_particles; i++) {
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = 1;
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/
	default_random_engine gen;
	const double EPSILON = 0.000001;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);	
	for (int i = 0; i < num_particles; i++) {
		double theta = particles[i].theta;
		if ( fabs(yaw_rate) >=EPSILON) {
			particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate*delta_t) - sin(theta)) + dist_x(gen);
			particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate*delta_t)) + dist_y(gen);
		} else {
			particles[i].x += velocity * delta_t * cos(theta) + dist_x(gen);
			particles[i].y += velocity * delta_t * sin(theta) + dist_y(gen);
		}
		particles[i].theta += yaw_rate * delta_t + dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	for(int i = 0; i < observations.size(); i ++) {
		double min_dist;
		int id;
		for ( int j = 0; j < predicted.size(); j ++) {
			double distance_ij = dist( observations[i].x, observations[i].y, predicted[j].x, predicted[j].y );
			if (( j == 0)|| (distance_ij < min_dist)) {
				//id = predicted[j].id;
				// save index only, not id
				id = j;
				min_dist = distance_ij;
			}
		}
		observations[i].id = id;
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
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
	weights.clear();
	for(int i = 0; i < num_particles; i ++) {
		Particle p = particles[i];

		// find all landmark within sensor range respective to this particle
		std::vector<LandmarkObs> predicted;
		for(int k = 0; k < map_landmarks.landmark_list.size(); k++) {
			Map::single_landmark_s landmark = map_landmarks.landmark_list[k];

			if ( dist(p.x, p.y, landmark.x_f, landmark.y_f ) <= sensor_range) {
				LandmarkObs obj;

				obj.x = landmark.x_f;
				obj.y = landmark.y_f;
				obj.id = landmark.id_i;
				predicted.push_back(obj);
			}
		}
		// convert observation coordinate to map coordinate
		std::vector<LandmarkObs> transformed_observation;
		for ( int j = 0; j < observations.size(); j ++) {
			// convert cooridnate according to particle theta and x, y
			LandmarkObs transformed_obs;

			transformed_obs.x = p.x + observations[j].x * cos(p.theta) - observations[j].y * sin(p.theta);
			transformed_obs.y = p.y + observations[j].x * sin(p.theta) + observations[j].y * cos(p.theta);
			transformed_observation.push_back(transformed_obs);
		}
		// Association
		dataAssociation(predicted, transformed_observation);
		//
		std::vector<int> associations;
		std::vector<double> sense_x;
		std::vector<double> sense_y;
		double std_x = std_landmark[0];
		double std_y = std_landmark[1];
		p.weight = 1.0;
		for ( int k = 0; k < transformed_observation.size(); k++ ) {
			// extract id from index
			double x = transformed_observation[k].x;
			double y = transformed_observation[k].y;
			int index = transformed_observation[k].id;
			int id = predicted[index].id;

			associations.push_back(id);
			sense_x.push_back(transformed_observation[k].x);
			sense_y.push_back(transformed_observation[k].y);

			// update weight of particle
			double ux = predicted[index].x;
			double uy = predicted[index].y;
			// calculate probability
			double probability = 1 / (2 * M_PI * std_x * std_y) * exp(-((x-ux)*(x-ux)/(2*std_x*std_x) + (y-uy)*(y-uy)/(2*std_y*std_y)));
			p.weight *= probability;
		}
		particles[i].weight = p.weight;
		weights.push_back(p.weight);
		SetAssociations( p, associations, sense_x, sense_y);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	default_random_engine gen;
	discrete_distribution<> dd(weights.begin(), weights.end());
	std::vector<Particle> new_particles;
	// resample usingin discrete distribution
	for (int i = 0; i < num_particles; i ++) {
		int index = dd(gen);
		Particle p1 = particles[index];
		new_particles.push_back(p1);
	}
	// replace particles
	particles.clear();
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

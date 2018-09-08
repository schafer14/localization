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
#include <float.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	default_random_engine gen;

  num_particles = 100;
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];

  normal_distribution<double> dist_x(x, std_x);
  normal_distribution<double> dist_y(y, std_y);
  normal_distribution<double> dist_theta(theta, std_theta);

  for (int i = 0; i < num_particles; i++) {
    Particle particle = {
      i,
      dist_x(gen),
      dist_y(gen),
      dist_theta(gen),
      1
    };

    particles.push_back(particle);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	default_random_engine gen;

  normal_distribution<double> dist_x(0, std_pos[0]);
  normal_distribution<double> dist_y(0, std_pos[1]);
  normal_distribution<double> dist_theta(0, std_pos[2]);

  for (int i = 0; i < num_particles; i ++) {
    double x, y, theta, new_x, new_y, new_theta;

    Particle particle = particles[i];
    x = particle.x;
    y = particle.y;
    theta = particle.theta;

    // Calculate without adding random noise
    if (yaw_rate != 0) {
      new_x = x + (velocity / yaw_rate) * (sin(theta + yaw_rate * delta_t) - sin(theta));
      new_y = y + (velocity / yaw_rate) * (cos(theta) - cos(theta + yaw_rate * delta_t));
      new_theta = theta + yaw_rate * delta_t;
    } else {
      new_x = x + (velocity * delta_t * cos(theta));
      new_y = y + (velocity * delta_t * sin(theta));
      new_theta = theta;
    }

    // Add random noise
    particles[i].x = new_x + dist_x(gen);
    particles[i].y = new_y + dist_y(gen);
    particles[i].theta = new_theta + dist_theta(gen);
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
	
	for (int i = 0; i < observations.size(); i ++) {
	  double min_dist = DBL_MAX;
	  int neighbor_id;
    LandmarkObs observation = observations[i];

	  for (int j = 0; j < predicted.size(); j++) {
	    LandmarkObs pred = predicted[j];
	    double distance = dist(pred.x, pred.y, observation.x, observation.y);

	    if (distance < min_dist) {
	      min_dist = distance;
	      neighbor_id = pred.id;
	    }
	  }

	  observations[i].id = neighbor_id;
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
  	const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	
	for (int i = 0; i < num_particles; i++) {
	  Particle particle = particles[i];
    std::vector<LandmarkObs> preds;
    std::vector<LandmarkObs> transformations;

	  for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
	    double landmark_x = map_landmarks.landmark_list[j].x_f;
	    double landmark_y = map_landmarks.landmark_list[j].y_f;
	    int landmark_id = map_landmarks.landmark_list[j].id_i;

	    if (fabs(landmark_x - particle.x) <= sensor_range && fabs(landmark_y - particle.y) <= sensor_range) {
	      preds.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y});
	    }
	  }


	  // Transformation
    for (int j = 0; j < observations.size(); j++) {
      LandmarkObs observation = observations[j];

      double tx = cos(particle.theta) * observation.x - sin(particle.theta) * observation.y + particle.x;
      double ty = sin(particle.theta) * observation.x + cos(particle.theta) * observation.y + particle.y;

      transformations.push_back(LandmarkObs{observation.id, tx, ty});
    }
	
	  dataAssociation(preds, transformations);
	  
	  particles[i].weight = 1;

    for (int j = 0; j < transformations.size(); j++) {
      LandmarkObs prediction;
      LandmarkObs observation = transformations[j];

      for (int k = 0; k < preds.size(); k++) {
        if (preds[k].id == transformations[j].id) {
          prediction = preds[k];
          break;
        }
      }

      double weight = (1 / (2 * M_PI * std_landmark[0]* std_landmark[1])) * exp(-(pow(prediction.x - observation.x, 2) / (2 * pow(std_landmark[0], 2)) + (pow(prediction.y - observation.y, 2)/(2 * pow(std_landmark[1], 2)))));

      particles[i].weight = particle.weight * weight;
    }
	}
}

void ParticleFilter::resample() {
  vector<Particle> new_particles;
	default_random_engine gen;

  vector<double> weights;
  for (int i = 0; i < num_particles; i++) {
    Particle particle = particles[i];
    weights.push_back(particle.weight);
  }

  double beta = 0;
  double max_weight = *max_element(weights.begin(), weights.end());

  uniform_int_distribution<int> int_dist(0, num_particles - 1);
  int index = int_dist(gen);

  uniform_real_distribution<double> real_dist(0, max_weight);

  for (int i = 0; i < num_particles; i++) {
    beta = beta + weights[index];
    while (beta > weights[index]) {
      beta = beta - weights[index];
      index = (index + 1) % num_particles;
    }
  
    new_particles.push_back(particles[index]);
  }

  particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates


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

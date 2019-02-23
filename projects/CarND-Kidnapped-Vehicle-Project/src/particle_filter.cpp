/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <limits>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::uniform_int_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  // ParticleFilter::num_particles = num_particles;

  std::default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  
  normal_distribution<double> dist_y(y, std[1]);
    
  normal_distribution<double> dist_theta(theta, std[2]);

  ParticleFilter::particles.clear();
  ParticleFilter::weights.clear();

  for (int i = 0; i < num_particles; ++i) {
    double sample_x, sample_y, sample_theta;

    sample_x = dist_x(gen);
    sample_y = dist_y(gen);
    sample_theta = dist_theta(gen);

    Particle p;
    p.id = i;
    p.x = sample_x;
    p.y = sample_y;
    p.theta = sample_theta;
    p.weight = 1;

    ParticleFilter::particles.push_back(p);

    ParticleFilter::weights.push_back(1);
  }

  ParticleFilter::is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::default_random_engine gen;

  for (int i = 0; i < ParticleFilter::num_particles; ++i) {
    if (yaw_rate != 0) {
      particles[i].x = particles[i].x + (velocity / yaw_rate) * (sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
      particles[i].y = particles[i].y + (velocity / yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
      particles[i].theta = particles[i].theta + yaw_rate * delta_t;
    } else {
      particles[i].x = particles[i].x + velocity * delta_t * cos(particles[i].theta);
      particles[i].y = particles[i].y + velocity * delta_t * sin(particles[i].theta);
    }
    

    normal_distribution<double> dist_x(particles[i].x, std_pos[0]);
    normal_distribution<double> dist_y(particles[i].y, std_pos[1]);
    normal_distribution<double> dist_theta(particles[i].theta, std_pos[2]); 

    particles[i].x = dist_x(gen);
    particles[i].y = dist_y(gen);
    particles[i].theta = dist_theta(gen); 

  }


}

double multiv_prob(double sig_x, double sig_y, double x_obs, double y_obs,
                   double mu_x, double mu_y) {
  // calculate normalization term
  double gauss_norm;
  gauss_norm = 1 / (2 * M_PI * sig_x * sig_y);

  // calculate exponent
  double exponent;
  exponent = (pow(x_obs - mu_x, 2) / (2 * pow(sig_x, 2)))
               + (pow(y_obs - mu_y, 2) / (2 * pow(sig_y, 2)));
    
  // calculate weight using normalization terms and exponent
  double weight;
  weight = gauss_norm * exp(-exponent);
    
  return weight;
}



void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  for (int i = 0; i < num_particles; ++i) {

    double current_weight = 1.0;

    for (int k = 0; k < observations.size(); ++k) {

      double temp_x = particles[i].x + cos(particles[i].theta) * observations[k].x - sin(particles[i].theta) * observations[k].y;

      double temp_y = particles[i].y + sin(particles[i].theta) * observations[k].x + cos(particles[i].theta) * observations[k].y;

      double dst = 1.79769e+308;
      int index = 0;

      for (int j = 0; j < map_landmarks.landmark_list.size(); ++j) {

        double current_dst;

        if (dist(map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f, particles[i].x, particles[i].y) <= sensor_range) {
          current_dst = dist(temp_x, temp_y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f);
        } else {
          // continue;
          current_dst = 1e+10;
        }
        if (current_dst < dst) {
          dst = current_dst;
          index = j;
        }
      }

      double temp = multiv_prob(std_landmark[0], std_landmark[1], 
                                  temp_x, temp_y,
                                  map_landmarks.landmark_list[index].x_f, map_landmarks.landmark_list[index].y_f);
      if (temp == 0) {
        temp = 1e-10;
      }

      current_weight *= temp;
    }

    particles[i].weight = current_weight;
    weights[i] = particles[i].weight;
    
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::random_device rd;

  vector<double> arr_w;
  
  for (int i = 0; i < ParticleFilter::num_particles; ++i) {
    arr_w.push_back(ParticleFilter::weights[i]);
  }

  std::discrete_distribution<> d(arr_w.begin(), arr_w.end());

  vector<Particle> new_particles;
  ParticleFilter::weights.clear();
  for(int n = 0; n < ParticleFilter::num_particles; ++n) {
    int temp = d(rd);
    new_particles.push_back(ParticleFilter::particles[temp]);
    ParticleFilter::weights.push_back(ParticleFilter::particles[temp].weight);
  }

  ParticleFilter::particles = new_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
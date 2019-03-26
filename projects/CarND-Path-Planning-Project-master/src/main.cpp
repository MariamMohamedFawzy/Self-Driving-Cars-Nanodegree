#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"


// for convenience
using nlohmann::json;
using std::string;
using std::vector;

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  string vehicle_status = "KL";
  double factor = 1.0;

  double max_speed = 21.0; // meters per second
  double max_acc = 10; // meters ** 2 per sec

  double speed = 15.0;
  double last_speed = 0.0;
  double acc = 5.0;

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
               &map_waypoints_dx,&map_waypoints_dy, &vehicle_status,
                &factor,
                 &max_speed ,&speed, &acc, &last_speed]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {
      // std::cout << "enter" << std::endl;
      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];
          car_speed *= 0.44704;
          // if (abs(car_speed - last_speed) > acc) {
          //   car_speed = last_speed;
          // }
          car_speed = last_speed;

          speed = car_speed;

          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side 
          //   of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];

          json msgJson;

          vector<double> next_x_vals;
          vector<double> next_y_vals;

          /**
           * TODO: define a path made up of (x,y) points that the car will visit
           *   sequentially every .02 seconds
           */

          if (previous_path_x.size() > 2) {
            car_s = end_path_s;
            // car_d = end_path_d;
          }

          
           int current_lane = car_d / 4;


          bool change_lane = false;
          bool change_speed = false;
          bool critic = true;
          std::vector<double> distances;
          distances.push_back(1000);
          distances.push_back(1000);
          distances.push_back(1000);

          double min_distance = 1000;

          double min_speed = car_speed;

          // std::cout << "speed " << speed << std::endl;


          for (int i = 0; i < sensor_fusion.size(); ++i) {

            vector<double> vehicle_info = sensor_fusion[i];
            int lane_num;

            if (vehicle_info[6] >= 0) {

              lane_num = vehicle_info[6] / 4;

              double distance = vehicle_info[5] - car_s;

              if (lane_num == current_lane) {

                if (distance >= 0 && distance < 30 && distance < min_distance) {

                  min_distance = distance;

                  double current_car_speed = sqrt(vehicle_info[3]*vehicle_info[3] + vehicle_info[4]*vehicle_info[4]);

                  min_speed = current_car_speed;

                  if (current_car_speed > 1 && current_car_speed + 3 < max_speed && current_car_speed < speed) {
                    change_speed = true;
                  } else {
                    change_lane = true; 
                  }

                }
              }

              if (distances[lane_num] > abs(distance)) {
                  distances[lane_num] = distance;
              }

              // for (int j = 0; j < 3; ++j) {
              //   if (lane_num == j && distances[j] > abs(distance)) {
              //     distances[j] = distance;
              //   }
              // }

            }
          }

          int new_lane = current_lane;

          if (change_speed || change_lane) {
            vector<int> possible_lanes;
            if (current_lane - 1 >= 0) {
              possible_lanes.push_back(current_lane - 1);
            }
            if (current_lane + 1 <= 2) {
              possible_lanes.push_back(current_lane + 1);
            }

            double max_distance = distances[current_lane];
            // new_lane = current;

            for (int i = 0; i < possible_lanes.size(); ++i) {
              if (distances[possible_lanes[i]] >= max_distance && distances[possible_lanes[i]] >= 30) {
                max_distance = distances[possible_lanes[i]];
                new_lane = possible_lanes[i];
              }
            }

            // if (distances[0] >= distances[1] && distances[0] >= distances[2]) {
            //   new_lane = 0;
            // } else if (distances[1] >= distances[0] && distances[1] >= distances[2]) {
            //   new_lane = 1;
            // } else {
            //   new_lane = 2;
            // }
            // if (abs(current_lane-new_lane) != 1 || abs(distances[new_lane]) <= 25) {
            //   change_lane = false;
            //   new_lane = current_lane;
            // }

            if (new_lane == current_lane) {
              // speed = min_speed;
              change_lane = false;
              change_speed = true;
            }
          } 



          // if (change_speed || change_lane) {
          //   factor = -1.0;
          // }

      
          double pos_x;
          double pos_y;
          double angle;
          int path_size = previous_path_x.size();

          std::vector<double> X_fre, Y_fre;

          if (path_size < 2) {

            double prev_x = car_x - cos(car_yaw);
            double prev_y = car_y - sin(car_yaw);

            X_fre.push_back(prev_x);
            X_fre.push_back(car_x);

            Y_fre.push_back(prev_y);
            Y_fre.push_back(car_y);

            pos_x = car_x;
            pos_y = car_y;
            angle = deg2rad(car_yaw);
            
          }
           else {
              
            pos_x = previous_path_x[path_size-1];
            pos_y = previous_path_y[path_size-1];

            double pos_x2 = previous_path_x[path_size-2];
            double pos_y2 = previous_path_y[path_size-2];
            angle = atan2(pos_y-pos_y2,pos_x-pos_x2);

            X_fre.push_back(pos_x2);
            X_fre.push_back(pos_x);

            Y_fre.push_back(pos_y2);
            Y_fre.push_back(pos_y);

          }

          tk::spline s;

          if (change_lane){
            std::cout << "current lane " << current_lane << " new lane " << new_lane << " speed " << speed << std::endl;
          }

          vector<double> xy_path;

          for (int i = 1; i <= 3; i++) {
            xy_path = getXY(car_s + i*30, new_lane*4+2,
                                          map_waypoints_s, 
                                          map_waypoints_x, 
                                          map_waypoints_y);

            X_fre.push_back(xy_path[0]);
            Y_fre.push_back(xy_path[1]);
          } 

          for (int i = 0; i < X_fre.size(); ++i) {
            double temp_x = X_fre[i];
            double temp_y = Y_fre[i];
            X_fre[i] = (temp_x-pos_x)*cos(0-angle) - (temp_y-pos_y) * sin(0-angle);
            Y_fre[i] = (temp_x-pos_x)*sin(0-angle) + (temp_y-pos_y) * cos(0-angle);
          } 
   

          s.set_points(X_fre,Y_fre);

          for (int i = 0; i < path_size; ++i) {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          double distance_x = 30.0;
          double distance_y = s(distance_x);
          double distance_dig = sqrt(distance_x*distance_x + distance_y*distance_y);

          

          double distance_x_so_far = 0;

          for (int i = 0; i < 50-path_size; ++i) {

            if (true) {
              if (change_lane || change_speed) {
                acc = -4.0;
              } else {
                acc = 5.0;
              }
              if (speed + factor * acc*0.02 <= max_speed && speed + factor * acc*0.02 >= min_speed) {
                speed = speed + factor * acc*0.02;
              }
            }


            int N = distance_dig / (0.02 * speed);
            double x = distance_x_so_far + distance_x/N;
            double y = s(x);

            distance_x_so_far = x;

            next_x_vals.push_back(pos_x + (x * cos(angle) - y * sin(angle)));
            next_y_vals.push_back(pos_y + (x * sin(angle) + y * cos(angle)));
          }

          last_speed = speed;
          // factor = 1.0;
          // change_lane = false;
          // critic = false;
          // change_speed = false;

          
       
          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}
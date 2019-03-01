#include <iostream>
#include <vector>
#include <math.h>

#include "PID.h"


/**
 * TODO: Complete the PID class. You may add any additional desired functions.
 */

PID::PID() {}

PID::~PID() {}

void PID::Init(double Kp_, double Ki_, double Kd_) {
  /**
   * TODO: Initialize PID coefficients (and errors, if needed)
   */
  Kp = Kp_;
  Ki = Ki_;
  Kd = Kd_;

  p_error = 0.0;
  d_error = 0.0;
  i_error = 0.0;

  // prev = 0.0;
  // int_cte = 0;
  // diff_cte = 0;
  // first = true;
}

void PID::UpdateError(double cte) {
  /**
   * TODO: Update PID errors based on cte.
   */
  d_error =cte - p_error;
  p_error = cte;
  if (prev.size() < 100) {
    prev.push_back(cte);
    i_error = 0;
  } else {
    prev.erase(prev.begin());
    prev.push_back(cte);
    i_error = 0;
    for (int i = 0; i < prev.size(); ++i) {
      i_error += prev[i];
    }
  }
  
  // i_error += cte;
}

double PID::TotalError() {
  /**
   * TODO: Calculate and return the total error
   */
  return  -Kd * d_error - Kp * p_error - Ki * i_error;  // TODO: Add your total error calc here!
}


// double run(std::vector<double> params, double cte, double n, double speed) {
//     double err = 0.0;
//     double prev_cte = cte;
//     double int_cte = 0;
//     for (int i = 0; i <(2 * n); ++i){
//         // cte = robot.y
//         double diff_cte = cte - prev_cte;
//         int_cte += cte;
//         prev_cte = cte;
//         double steer = -params[0] * cte - params[1] * diff_cte - params[2] * int_cte;
//         if (steer > 1) {
//           steer = 1;
//         } else if (steer < -1) {
//           steer = -1;
//         }
//         // robot.move(steer, speed)
//         if (i >= n)
//             err += cte * cte;
//         //
//         cte = cte - cos(steer) * speed;
//     }
//     return err / n;
// }

// void PID::Twidle(double tolerance) {

//   std::vector<double> dp;  
//   dp.push_back(1);
//   dp.push_back(1);
//   dp.push_back(1);

//   std::vector<double> p;  
//   p.push_back(Kp);
//   p.push_back(Kd);
//   p.push_back(Ki);

//   double best_err = run(p, 1, 100, 1.0);

//   // std::cout << "enter" << tolerance << std::endl;
//   while(dp[0] + dp[1] + dp[2] > tolerance) {
//     for(int i = 0; i < p.size(); ++i) {
//       p[i] += dp[i];
//       double err = run(p, 1, 100, 1.0); // cte = 1, speed = 1.0


//       if (err < best_err){
//         best_err = err;
//         dp[i] *= 1.1;
//       }
//       else {
//         p[i] -= 2*dp[i];
//         err = run(p, 1, 100, 1.0); 
 
//         if (err < best_err) {
//           best_err = err;
//           dp[i] *= 1.1;
//         }
//         else {
//           p[i] += dp[i];
//           dp[i] *= 0.9;
//         }
 
//       }

//     }
//   }

//   Kp = p[0];
//   Kd = p[1];
//   Ki = p[2];

//   std::cout << "Kp: " << Kp << " Kd: " <<  Kd << " Ki: " << Ki
//                     << std::endl;

// }
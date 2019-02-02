#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;
using std::cout;
using std::endl;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * Calculate the RMSE here.
   */
   VectorXd rmse(4);
   rmse << 0,0,0,0;

   for (unsigned int i=0; i < estimations.size(); ++i) {
      VectorXd residual = estimations[i] - ground_truth[i];
      residual = residual.array()*residual.array();
      rmse += residual;
   }

   rmse = rmse/estimations.size();
   rmse = rmse.array().sqrt();

  
   return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  /**
   * Calculate a Jacobian here.
   */
  MatrixXd Hj(3,4);
  // recover state parameters
  float px = x_state(0);
  float py = x_state(1);
  float vx = x_state(2);
  float vy = x_state(3);

      float px_2_py_2 = px*px + py*py;
      if (px_2_py_2 < 0.00001) {
         px = 0.001;
         py = 0.001;
         px_2_py_2 = px*px + py*py;
      }

      Hj << px/sqrt(px_2_py_2), py/sqrt(px_2_py_2), 0, 0,
            -py/px_2_py_2, px/px_2_py_2, 0, 0,
            py*(vx*py - vy*px)/pow(px_2_py_2, 1.5), 
            px*(vy*px - vx*py)/pow(px_2_py_2, 1.5), 
            px/sqrt(px_2_py_2), py/sqrt(px_2_py_2) ;     
  return Hj;
}

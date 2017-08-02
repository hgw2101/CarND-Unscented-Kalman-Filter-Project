#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */

  // copy directly from EKF in project 1 since all parameters are the same
  // the main.cpp file already calculated the v_x and v_y estimates based 
  // on v and yaw values
  VectorXd rmse(4);
  rmse<<0,0,0,0; // initialize rmse vector with all 0's

  // check validity of input data
  if(estimations.size() != ground_truth.size() || estimations.size() == 0) {
    cout<<"Invalid estimation or ground_truth data"<<endl;
    return rmse;
  }

  // iterate through the estimations and ground_truth arrays
  for(unsigned int i=0; i < estimations.size(); ++i) {

    // calculate residual
    VectorXd residual = estimations[i] - ground_truth[i];
    residual = residual.array()*residual.array(); // square it

    rmse += residual;
  }

  // calculate the mean
  rmse = rmse/estimations.size();

  // take square root
  rmse = rmse.array().sqrt();

  return rmse;

}
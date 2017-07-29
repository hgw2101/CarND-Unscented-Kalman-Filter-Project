#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // define state dimension
  int n_x_  = 5;

  // define augmented state dimension, // n_x_ plus process noise, i.e. acceleration noise for longitudinal and yaw
  int n_aug_ = 7; 

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  // TODO: need to fine-tune the values here!!!
  // P_ = MatrixXd(n_x_, n_x_);
  P_ << 1,0,0,0,0,
        0,1,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,0,0,0,1;

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // weights of sigma points
  weights_ = VectorXd(2*n_x_+1);

  // define spreading parameter, lambda
  double lambda = 3 - n_x_;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */


}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  //**Step 1: calculate sigma points

  // initialize the Xsig_aug matrix, the temporary matrix used to hold the augmented sigma points,
  // the dimension of this is n_aug_ by 2*n_aug_+1, the end result, Xsig_pred_ will be a
  // n_x_ by 2*n_aug_+1 matrix, since you don't predict values for the process noise parameters.
  // Also, initialize the x_aug vector for convenience
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2*n_aug_+1);
  
  VectorXd x_aug = VectorXd(n_aug_);
  x_aug.head(n_x_) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //set first column of sigma point matrix to the augmented state vector
  Xsig_aug.col(0) = x_aug;

  //initialize the augmented process covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  // square root of P_aug
  MatrixXd A = P_aug.llt().matrixL();

  //set the remaining sigma points
  for (int i=0; i<2*n_aug_+1; i++) {
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda+n_aug_) * A.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda+n_aug_) * A.col(i);
  }

  //**Step 2: predict sigma points using the process model
  // simply take each sigma point, i.e. each column of the Xsig_aug matrix, and run it through the process model plus noise
  
  for (int i=0; i<2*n_aug_+1; i++) {
    //extract values for better readability
    double p_x = Xsig_aug(0,i);
    double p_y = Xsig_aug(1,i);
    double v = Xsig_aug(2,i);
    double yaw = Xsig_aug(3,i);
    double yawd = Xsig_aug(4,i);
    double nu_a = Xsig_aug(5,i);
    double nu_yawdd = Xsig_aug(6,i);

    //predicted state values for the sigma points
    double px_p, py_p;
    if (fabs(yawd) > 0.001) {
      px_p = p_x + v/yawd * (sin(yaw + yawd*delta_t) - sin(yaw)) + delta_t*delta_t*cos(yaw)*nu_a/2;
      py_p = p_y + v/yawd * (-1*cos(yaw + yawd*delta_t) + cos(yaw)) + delta_t*delta_t*sin(yaw)*nu_a/2;
    } else {
      px_p = p_x + v * cos(yaw) * delta_t + delta_t*delta_t*cos(yaw)*nu_a/2;
      py_p = p_y + v * sin(yaw) * delta_t + delta_t*delta_t*sin(yaw)*nu_a/2;
    }

    double v_p = delta_t * nu_a;
    double yaw_p = yawd * delta_t + delta_t*delta_t*nu_yawdd/2;
    double yawd_p = delta_t * nu_yawdd;

    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  //**Step 3: using predicted sigma points to predict the new state mean vector and state covariance

  //weights for each sigma point
  weights_(0) = lambda / (lambda + n_aug_);
  for (int i=1; i<2*n_aug_+1; i++) {
    weights_(i) = 1 / (2 * (lambda + n_aug_));
  }

  //predicted state mean vector
  for (int i=0; i<2*n_aug_; i++) {
    x += weights_(i) * Xsig_pred_.col(i);
  }

  //predicted state covariance matrix
  for (int i=0; i<2*n_aug_; i++) {
    P += (weights_(i) * (Xsig_pred_.col(i) - x)) * (Xsig_pred_.col(i) - x).transpose();
    // P += (weights (i) * (Xsig_pred .col(i) - x)) * (Xsig_pred .col(i) - x).transpose();
    // P += (weights (i) * (Xsig_pred .col(i) - x)) * (Xsig_pred .col(i) - x).transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}

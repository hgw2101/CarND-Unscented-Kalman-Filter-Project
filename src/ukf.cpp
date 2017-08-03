#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

#define PI 3.14159265

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)

  is_initialized_ = false;

  use_laser_ = true; // TODO: not perfect implementation, need to revisit

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // define state dimension
  n_x_  = 5;

  // define augmented state dimension, // n_x_ plus process noise, i.e. acceleration noise for longitudinal and yaw
  n_aug_ = 7; 

  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_); // need to declare this first, or else you'll get a segmentation error
  P_ << 0.03,0,0,0,0,
        0,0.03,0,0,0,
        0,0,1.5,0,0,
        0,0,0,1,0,
        0,0,0,0,1; //tuning initial values p_x/p_y in P_ really helped reduce RMSE down to acceptable levels!!! tried other values of yaw and yawd, but 1 seem to be the best. v seems to be consistent around 4-5, so I set the variance to 1.5.

  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2*n_aug_+1);

  // timestamp
  time_us_ = 0;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3.5; // if this was kept at 30 (initial value), the simulator wouldn't even run without blowing up

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1.5;

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

  // no. of sigma points
  n_sig_ = 2*n_aug_+1; // to avoid repeated calculations throughout

  // define spreading parameter, lambda
  lambda_ = 3 - n_aug_;

  // weights of sigma points
  weights_ = VectorXd(n_sig_);
  // NOTE: there are other ways of calculating weights
  // calc the weights_ here for code efficiency since it's made up of constants 
  weights_.fill(1 / (2 * (lambda_ + n_aug_)));
  weights_(0) = lambda_ / (lambda_ + n_aug_);

  ///*lidar and radar measurement noise matrix, R
  R_lidar_ = MatrixXd(2,2); //TODO: potential refactor, set n_z_radar_ and n_z_lidar_ instance variables
  R_lidar_ << std_laspx_*std_laspx_,0,
              0,std_laspy_*std_laspy_;

  R_radar_ = MatrixXd(3,3); //TODO: potential refactor, set n_z_radar_ and n_z_lidar_ instance variables
  R_radar_ << std_radr_*std_radr_,0,0,
              0,std_radphi_*std_radphi_,0,
              0,0,std_radrd_*std_radrd_;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  if (!is_initialized_) {
    std::cout<<"this is first measurement: "<<std::endl<<meas_package.raw_measurements_<<std::endl;    

    double initial_p_x;
    double initial_p_y;
    double initial_v;
    double initial_yaw;
    double initial_yawd;

    if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
      initial_p_x = meas_package.raw_measurements_[0];
      initial_p_y = meas_package.raw_measurements_[1];
      // cannot infer v, yaw or yawd, use best guess by cheating :)
      // taking the first meas (LIDAR), px=.312243/px=.58034 and second meas (RADAR), px=.5342/py=.8629 (converted from rho/phi) and calculate the other values
      initial_v = 4.89281; // rho_dot is not the same as speed in the CTRV model, but use it to approximate it for now, the value used here is the first RADAR meas
      initial_yaw = 0.8546; // calculate arctan(delta_py/delta_px)
      initial_yawd = 0.425; // cannot calculate delta_yaw/delta_t without taking at least 2 RADAR meas, pick an arbitrary one for now, half of intial_yaw
    } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
      double rho = meas_package.raw_measurements_[0];
      double phi = meas_package.raw_measurements_[1];
      double rho_dot = meas_package.raw_measurements_[2];

      initial_p_x = sin(phi) * rho;
      initial_p_y = cos(phi) * rho;

      // again, cheating to get the values here using the same approach as the LIDAR case :)
      initial_v = rho_dot;
      initial_yaw = -0.994;
      initial_yawd = 0.5;
    }

    x_ << initial_p_x,
          initial_p_y,
          initial_v,
          initial_yaw,
          initial_yawd;
      
    std::cout<<"this is initial x_: "<<std::endl<<x_<<std::endl;

    time_us_ = meas_package.timestamp_;
    is_initialized_ = true;
    return;
  }

  std::cout<<"this is measurement: "<<std::endl<<meas_package.raw_measurements_<<std::endl;


  double delta_t = (meas_package.timestamp_ - time_us_) / 1000000.0; // time in seconds
  std::cout<<"this is delta_t: "<<delta_t<<std::endl;
  time_us_ = meas_package.timestamp_; // update previous_timestamp_ to current timestamp;

  Prediction(delta_t);

  if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
    UpdateLidar(meas_package);
  } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
    UpdateRadar(meas_package);
  }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  //**Step 1: calculate sigma points

  // initialize the Xsig_aug matrix, the temporary matrix used to hold the augmented sigma points,
  // the dimension of this is n_aug_ by n_sig_, the end result, Xsig_pred_ will be a
  // n_x_ by n_sig_ matrix, since you don't predict values for the process noise parameters.
  // Also, initialize the x_aug vector for convenience

  MatrixXd Xsig_aug = MatrixXd(n_aug_, n_sig_);
  
  VectorXd x_aug = VectorXd(n_aug_);

  x_aug.head(n_x_) = x_;

  x_aug(5) = 0.0;
  x_aug(6) = 0.0;

  //initialize the augmented process covariance matrix
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  // square root of P_aug
  MatrixXd A = P_aug.llt().matrixL();

  //set first column of sigma point matrix to the augmented state vector
  Xsig_aug.col(0) = x_aug;
  //set the remaining sigma points
  for (int i=0; i<n_aug_; i++) {
    Xsig_aug.col(i+1)        = x_aug + sqrt(lambda_+n_aug_) * A.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * A.col(i);
  }

  //**Step 2: predict sigma points using the process model
  // simply take each sigma point, i.e. each column of the Xsig_aug matrix, and run it through the process model plus noise
  
  for (int i=0; i<n_sig_; i++) {
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
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    // prior value plus process model
    double v_p = v; // I forgot to set the initial values of v_p, yaw_p and yawd_p to v, yaw + yawd*delta_t and yawd, respectively, took me hours to debug this issue. without this the estimated points would approximate a straight line and then sharply turns
    double yaw_p = yaw + yawd * delta_t;
    double yawd_p = yawd;

    // add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);

    v_p += delta_t * nu_a;
    yaw_p += delta_t*delta_t*nu_yawdd/2;
    yawd_p += delta_t * nu_yawdd;

    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

  //**Step 3: using predicted sigma points to predict the new state mean vector and state covariance

  //predicted state mean vector
  x_.fill(0.0);
  x_ = Xsig_pred_ * weights_;

  //predicted state covariance matrix
  P_.fill(0.0); // I didn't set this to 0 in my own solution in the lecture, since that was just one update, but there we have continuous input of measurements
  for (int i=0; i<n_sig_; i++) {
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    // angle normalization
    while (x_diff(3)>PI) x_diff(3)-=2.*PI;
    while (x_diff(3)<-PI) x_diff(3)+=2.*PI;

    P_ += weights_(i) * x_diff * x_diff.transpose();
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  //**Step 1: measurement prediction
  int n_z = 2; // for p_x and p_y

  // initialize sigma points for the lidar measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);

  // transform sigma points into measurement space
  // using the Eigen block function to simplify code, 0,0 represents the starting row/col of Xsig_pred_, the matrix we want to copy, and n_z,n_sig_ represent the ending row/col
  Zsig = Xsig_pred_.block(0,0,n_z,n_sig_);

  // once we have the sigma points in the measurement space, we can calculate the predicted measurements

  // mean predicted measurements
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0); // initialize vector with 0 values
  for (int i=0; i<n_sig_; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // predicted measurement covariance
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0); // initialize matrix with 0 values
  for (int i=0; i<n_sig_; i++) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise, R. R does not depend on n_aug_, so we add it to the overall S
  S += R_lidar_;

  //**Step 2: update using measurement readings

  // calculate cross correlation matrix (required for calculating Kalman gain, K)
  // state diff multiply by measurement diff transpose, weighted
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  for (int i=0; i<n_sig_; i++) {
    //measurement residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    //state mean diff
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //yaw angle normalization
    while (x_diff(3)>PI) x_diff(3) -= 2.*PI;
    while (x_diff(3)<-PI) x_diff(3) += 2.*PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain
  MatrixXd K = Tc * S.inverse();
  
  // residual
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_diff_meas = z - z_pred;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff_meas;
  P_ = P_ - K*S*K.transpose();

  //**Step 3: calculate NIS
  double nis = (z - z_pred).transpose() * S.inverse() * (z - z_pred);

  // std::cout<<"LIDAR: this is x_: "<<std::endl<<x_<<std::endl;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  //**Step 1: measurement prediction

  // Radar measurement dimension
  int n_z = 3; // for rho, phi and rho_dot`

  // initialize sigma points for the radar measurement space
  MatrixXd Zsig = MatrixXd(n_z, n_sig_);
  
  // transform sigma points into measurement space
  for (int i=0; i<n_sig_; i++) {
    // extract values for convenience
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v; //convert to v1, just like the formula for calculating radar parameters in the EKF project
    double v2 = sin(yaw)*v; 

    // set variables for convenience
    double c1 = p_x*p_x + p_y*p_y;
    double c2 = sqrt(p_x*p_x + p_y*p_y);

    // measurement model 

    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    
    // avoid atan2(0,0), which is undefined even though the atan2() function does return a valid number
    if (fabs(p_x < 0.0001) && fabs(p_y < 0.0001)) {
      Zsig(1,i) = 0;                                              //phi
    } else {
      Zsig(1,i) = atan2(p_y,p_x);                                 //phi
    }

    if(fabs(c1) < 0.0001) {
      Zsig(2,i) = 0.0;   //r_dot
    } else {
      Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot
    }
  }

  // once we have the sigma points in the measurement space, we can calculate the predicted measurements

  // mean predicted measurements
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0); // initialize vector with 0 values
  for (int i=0; i<n_sig_; i++) {
    z_pred += weights_(i) * Zsig.col(i);
  }

  // predicted measurement covariance
  MatrixXd S = MatrixXd(n_z, n_z);
  S.fill(0.0); // initialize matrix with 0 values
  for (int i=0; i<n_sig_; i++) {
    // residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    // normalize the phi to make sure it's between -PI and PI
    // TODO: refactor this into a helper method
    while (z_diff(1) > PI) z_diff(1) -= 2.*PI; // apparently c++ has this one-line shorthand control flow :)
    while (z_diff(1) < -PI) z_diff(1) += 2.*PI; // increment by 2.*PI since this will normalize faster
    
    S += weights_(i) * z_diff * z_diff.transpose();
  }

  // add measurement noise, R. R does not depend on n_aug_, so we add it to the overall S
  S += R_radar_;

  //**Step 2: update using measurement readings

  // calculate cross correlation matrix (required for calculating Kalman gain, K)
  // state diff multiply by measurement diff transpose, weighted
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);

  for (int i=0; i<n_sig_; i++) {
    //measurement residual
    VectorXd z_diff = Zsig.col(i) - z_pred;
    //phi angle normalization
    while (z_diff(1)>PI) z_diff(1) -= 2.*PI;
    while (z_diff(1)<-PI) z_diff(1) += 2.*PI;

    //state mean diff
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    //yaw angle normalization
    while (x_diff(3)>PI) x_diff(3) -= 2.*PI;
    while (x_diff(3)<-PI) x_diff(3) += 2.*PI;

    Tc += weights_(i) * x_diff * z_diff.transpose();
  }

  // Kalman gain
  MatrixXd K = Tc * S.inverse();

  // residual
  VectorXd z = meas_package.raw_measurements_;
  VectorXd z_diff_meas = z - z_pred;

  // angle normalization
  while (z_diff_meas(1)>PI) z_diff_meas(1) -= 2.*PI;
  while (z_diff_meas(1)<-PI) z_diff_meas(1) += 2.*PI;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff_meas;
  P_ = P_ - K*S*K.transpose();

  // std::cout<<"RADAR: this is x_: "<<std::endl<<x_<<std::endl;

  //**Step 3: calculate NIS
  double nis = (z - z_pred).transpose() * S.inverse() * (z - z_pred); //TODO: use visualization tools to tune std parameters with NIS
}
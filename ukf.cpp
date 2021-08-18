#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 10;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 10;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

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
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */

  is_initialized_ = false;
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  time_us_ = 0;
  Xsig_pred_ = MatrixXd(n_x_,2*n_aug_+1);
  weights_ = VectorXd(2*n_aug_+1);
  P_ = MatrixXd::Identity(5,5);
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */
  if(!is_initialized_){
    if(meas_package.sensor_type_ == meas_package.RADAR){
      double px = meas_package.raw_measurements_[0]*cos(meas_package.raw_measurements_[1]);
      double py = meas_package.raw_measurements_[0]*sin(meas_package.raw_measurements_[1]);
      x_ << px , py , 0, 0, 0;
      use_radar_ = true;
      use_laser_ = false;
    }
    else if(meas_package.sensor_type_ == meas_package.LASER){
      x_ << meas_package.raw_measurements_[0],meas_package.raw_measurements_[1],0,0,0;
      use_laser_ = true;
      use_radar_ = false;
    }
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
    return;
  }
  double delta_t = (meas_package.timestamp_ - time_us_)/1000000.0;
  time_us_ = meas_package.timestamp_;

  Prediction(delta_t);
  if(meas_package.sensor_type_ == meas_package.LASER && use_laser_){
    UpdateLidar(meas_package);
  }
  else if(meas_package.sensor_type_ == meas_package.RADAR && use_radar_){
    UpdateRadar(meas_package);
  }
  // print the output
  cout << "x_ = " << x_ << endl;
  cout << "P_ = " << P_ << endl;
}

void UKF::Prediction(double delta_t) {
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

  VectorXd x_aug(n_aug_);
  MatrixXd Q(2,2);
  Q << std_a_*std_a_ , 0,
       0,std_yawdd_*std_yawdd_;

  MatrixXd P_aug(n_aug_,n_aug_);
  MatrixXd Xsig_aug(n_aug_,2*n_aug_+1);

  x_aug.head(n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_+1) = 0;

  P_aug.topLeftCorner(n_x_,n_x_) = P_;
  P_aug.bottomRightCorner(2,2) = Q;

  MatrixXd A = P_aug.llt().matrixL();
  Xsig_aug.leftCols(1) = x_aug;

  for(int i = 0;i< n_aug_;i++){
    Xsig_aug.col(i+1) = x_aug + sqrt(lambda_+n_aug_)*A.col(i);
    Xsig_aug.col(i+n_aug_+1) = x_aug - sqrt(lambda_+n_aug_)*A.col(i);
  }
  cout << Xsig_aug << endl;
  VectorXd x_p(n_x_);

  for(int j=0;j<2*n_aug_+1;j++){
  double px = Xsig_aug(0,j);
  double py = Xsig_aug(1,j);
  double v = Xsig_aug(2,j);
  double phi = Xsig_aug(3,j);
  double phi_dot = Xsig_aug(4,j);
  double a = Xsig_aug(5,j);
  double phi_dd = Xsig_aug(6,j);
  double px_p , py_p;
  if(fabs(phi_dot) > 0.001){
    if(phi_dot ==0.0){
      px_p = 0.0;
      py_p = 0.0;
    }
    else{
      px_p = px + (v/phi_dot)*(sin(phi+phi_dot*delta_t)-sin(phi)) + 0.5*pow(delta_t,2)*cos(phi)*a;
      py_p = py + (v/phi_dot)*(-cos(phi+phi_dot*delta_t)+cos(phi)) + 0.5*pow(delta_t,2)*sin(phi)*a;
    }
  }
  else{
    px_p = px + v*cos(phi)*delta_t + 0.5*pow(delta_t,2)*cos(phi)*a;
    py_p = py + v*sin(phi)*delta_t + 0.5*pow(delta_t,2)*sin(phi)*a;
  }

  x_p << px_p , py_p ,v + delta_t*a, phi + phi_dot*delta_t+0.5*pow(delta_t,2)*phi_dd,phi_dot + delta_t*phi_dd;
  Xsig_pred_.col(j) = x_p;
} 
  
cout << Xsig_pred_ << endl;

weights_(0) = lambda_ / (lambda_+n_aug_);
for(int i =1;i<2*n_aug_+1;i++)
  weights_(i) = 1 / 2*(lambda_+n_aug_);

x_.fill(0.0);
P_.fill(0.0);
for(int i =0;i<2*n_aug_+1;i++)
  x_ += weights_(i)*Xsig_pred_.col(i);
  
for(int i =0;i<2*n_aug_+1;i++){
  VectorXd x_diff = (Xsig_pred_.col(i) - x_);
  while(x_diff(3) > M_PI) x_diff(3) -= 2.*M_PI;
  while(x_diff(3) < -M_PI) x_diff(3) += 2.*M_PI;
  P_ += weights_(i)*x_diff*x_diff.transpose();
}
cout << x_ << endl;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */
  int nz = 2;
  VectorXd z_(nz);
  VectorXd z(nz);
  z << meas_package.raw_measurements_[0],meas_package.raw_measurements_[1];
  MatrixXd Zsig_pred_(nz,2*n_aug_+1);
  MatrixXd S_(nz,nz);
  MatrixXd R(nz,nz);
  R << std_laspx_*std_laspx_,0,
       0,std_laspy_*std_laspy_;
  Zsig_pred_ = Xsig_pred_.topRows(2);
  cout <<"Zsig_pred = " << Zsig_pred_ << endl;
  z_.fill(0.0);
  S_.fill(0.0);
  for(int i=0;i<2*n_aug_+1;i++)
    z_ += weights_(i)*Zsig_pred_.col(i);
  
  for(int i=0;i<2*n_aug_+1;i++){
    VectorXd z_diff = (Zsig_pred_.col(i) - z_);
    S_ += weights_(i)*z_diff*z_diff.transpose();
  }
  S_ += R;

  MatrixXd T(n_x_,nz);
  T.fill(0.0);
  for(int i=0;i<2*n_aug_+1;i++){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig_pred_.col(i) - z_;
    T += weights_(i)*x_diff*z_diff.transpose();
  }
  MatrixXd K(n_x_,nz);
  K = T*S_.inverse();
  x_ = x_ + K*(z - z_);
  P_ = P_ - K*S_*K.transpose();
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */
  int nz = 3;
  VectorXd z_(nz);
  VectorXd z(nz);
  z << meas_package.raw_measurements_[0],meas_package.raw_measurements_[1],meas_package.raw_measurements_[2];
  MatrixXd Zsig_pred_(nz,2*n_aug_+1);
  MatrixXd S_(nz,nz);
  MatrixXd R(nz,nz);
  R << std_radr_*std_radr_,0,0,
       0,std_radphi_*std_radphi_,0,
       0,0,std_radrd_*std_radrd_; 
  VectorXd zsig_p(nz);
  Zsig_pred_.fill(0.0);
  for(int i=0;i<2*n_aug_+1;i++){
    double px = Xsig_pred_(0,i);
    double py = Xsig_pred_(1,i);
    double v = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double rho = sqrt(px*px+py*py);
    double phi = atan2(py,px);
    double rho_d;
    if(rho < 0.001)
      rho_d = 0.0;
    else
      rho_d = (px*v*cos(phi)+py*v*sin(phi))/rho;
    zsig_p << rho , phi , rho_d;
    Zsig_pred_.col(i) = zsig_p;
  }

  z_.fill(0.0);
  S_.fill(0.0);
  for(int i=0;i<2*n_aug_+1;i++)
    z_ += weights_(i)*Zsig_pred_.col(i);
  
  for(int i=0;i<2*n_aug_+1;i++){
    VectorXd z_diff = (Zsig_pred_.col(i) - z_);
    S_ += weights_(i)*z_diff*z_diff.transpose();
  }
  S_ += R;

  MatrixXd T(n_x_,nz);
  T.fill(0.0);
  for(int i=0;i<2*n_aug_+1;i++){
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    VectorXd z_diff = Zsig_pred_.col(i) - z_;
    T += weights_(i)*x_diff*z_diff.transpose();
  }
  MatrixXd K(n_x_,nz);
  K = T*S_.inverse();
  x_ = x_ + K*(z - z_);
  P_ = P_ - K*S_*K.transpose();
}

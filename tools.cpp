#include "tools.h"
#include <iostream>

using Eigen::VectorXd;
using std::vector;
using namespace std;


Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
   * TODO: Calculate the RMSE here.
   */
   VectorXd rmse(4);
  rmse = VectorXd::Zero(4);

  if(estimations.size() == 0){
     cout << "Estimation vector is empty " << endl;
     return rmse;
  }
  else if(estimations.size() != ground_truth.size()){
     cout << "Estimation vector size is not equal to ground truth vector size" << endl;
     return rmse;
  }

  for(int i=0;i<estimations.size();i++) {
     VectorXd sum = estimations[i] - ground_truth[i];
     sum = sum.array()*sum.array();
     rmse += sum;
  }


  rmse = rmse / estimations.size();
  rmse = rmse.array().sqrt();

  return rmse;
}
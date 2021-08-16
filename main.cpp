#include <iostream>
#include "../Eigen/Dense"
#include <vector>
#include "ukf.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

int main() {

	//Create a UKF instance
	UKF ukf;

    MatrixXd Xsig = MatrixXd(5, 15);
    ukf.SigmaPointPrediction(&Xsig);

    VectorXd x_pred(5);
    MatrixXd P_pred(5,5);
    ukf.PredictMeanAndCovariance(&x_pred,&P_pred);

    //print result
    std::cout << "Xsig = " << std::endl << Xsig << std::endl;

	return 0;
}
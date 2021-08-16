#ifndef UKF_H_
#define UKF_H_
#include "../Eigen/Dense"
#include <vector>

using namespace std;
using Eigen::VectorXd;
using Eigen::MatrixXd;

class UKF {

public: 


    UKF();

    //destructor
    virtual ~UKF();

    void Init();

void GenerateSigmaPoints(MatrixXd* Xsig_out);
void AugmentedSigmaPoints(MatrixXd* Xsig_out);
void SigmaPointPrediction(MatrixXd* Xsig_out);
void PredictMeanAndCovariance(VectorXd* x_pred, MatrixXd* P_pred);
};


#endif
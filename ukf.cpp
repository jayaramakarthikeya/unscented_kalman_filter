#include <iostream>
#include <cmath>
#include "ukf.h"

UKF::UKF(){
    Init();
}

UKF::~UKF(){

}

void UKF::Init() {

}

void UKF::GenerateSigmaPoints(MatrixXd* Xsig_out){

    //set state dimension
    int nx =  5;

    //define spreading parameter
    double lambda = 3 - nx;

    //set example state
    VectorXd x = VectorXd(nx);
    x <<   5.7441,
            1.3800,
            2.2049,
            0.5015,
            0.3528;

    //set example covariance matrix
    MatrixXd P = MatrixXd(nx, nx);
    P <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
            -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
            0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
            -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
            -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

    MatrixXd Xsig = MatrixXd(nx,nx*2 + 1);

    //calcuate square root of P
    MatrixXd A = P.llt().matrixL();
    Xsig.col(0) = x;

    for (int i=0;i<nx;i++){
        Xsig.col(i+1) = x + sqrt(lambda+nx)*A.col(i);
        Xsig.col(i+nx+1) = x - sqrt(lambda+nx)*A.col(i);
    }

    //cout << "Xsig = " << endl << Xsig << endl;

    *Xsig_out = Xsig;

}

void UKF::AugmentedSigmaPoints(MatrixXd* Xsig_out){

    int nx = 5;

    int naug = 7;

    double std_a = 0.2;
    double std_yawdd = 0.2;
    double lambda = 3 - naug;

    MatrixXd Q(2,2);
    Q << std_a*std_a , 0,
         0,std_yawdd*std_yawdd;

    //set example state
    VectorXd x = VectorXd(nx);
    x <<   5.7441,
            1.3800,
            2.2049,
            0.5015,
            0.3528;

    //create example covariance matrix
    MatrixXd P = MatrixXd(nx, nx);
    P <<     0.0043,   -0.0013,    0.0030,   -0.0022,   -0.0020,
            -0.0013,    0.0077,    0.0011,    0.0071,    0.0060,
            0.0030,    0.0011,    0.0054,    0.0007,    0.0008,
            -0.0022,    0.0071,    0.0007,    0.0098,    0.0100,
            -0.0020,    0.0060,    0.0008,    0.0100,    0.0123;

    VectorXd x_aug = VectorXd(nx+2);
    MatrixXd P_aug = MatrixXd(nx+2,nx+2);

    MatrixXd Xsig_aug = MatrixXd(naug,2*naug+1);

    x_aug.topRows(5) = x;
    x_aug(5) = 0;
    x_aug(6) = 0;

    P_aug.topLeftCorner(nx,nx) = P;
    P_aug.bottomRightCorner(2,2) = Q;

    MatrixXd A = P_aug.llt().matrixL();
    Xsig_aug.leftCols(1) = x_aug;

    for (int i=0;i<naug;i++){
        Xsig_aug.col(i+1) = x_aug + sqrt(lambda+naug)*A.col(i);
        Xsig_aug.col(i+naug+1) = x_aug - sqrt(lambda+naug)*A.col(i);
    }

    *Xsig_out = Xsig_aug;
    
}

void UKF::SigmaPointPrediction(MatrixXd* Xsig_out) {

    int nx = 5;
    int naug = 7;

    MatrixXd Xsig_aug(naug,2*naug+1);
    Xsig_aug << 5.7441, 5.85768,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.63052,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,   5.7441,
      1.38,  1.34566,  1.52806,     1.38,     1.38,     1.38,     1.38,     1.38,   1.41434,  1.23194,     1.38,     1.38,     1.38,     1.38,     1.38,
    2.2049,  2.28414,  2.24557,  2.29582,   2.2049,   2.2049,   2.2049,   2.2049,   2.12566,  2.16423,  2.11398,   2.2049,   2.2049,   2.2049,   2.2049,
    0.5015,  0.44339, 0.631886, 0.516923, 0.595227,   0.5015,   0.5015,   0.5015,   0.55961, 0.371114, 0.486077, 0.407773,   0.5015,   0.5015,   0.5015,
    0.3528, 0.299973, 0.462123, 0.376339,  0.48417, 0.418721,   0.3528,   0.3528,  0.405627, 0.243477, 0.329261,  0.22143, 0.286879,   0.3528,   0.3528,
         0,        0,        0,        0,        0,        0,  0.34641,        0,         0,        0,        0,        0,        0, -0.34641,        0,
         0,        0,        0,        0,        0,        0,        0,  0.34641,         0,        0,        0,        0,        0,        0, -0.34641;

    MatrixXd Xsig_pred(nx,2*naug+1);

    double delta_t = 0.1;

    for (int i = 0;i< 2*naug+1;i++) {

        double px = Xsig_aug(0,i);
        double py = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double phi = Xsig_aug(3,i);
        double phi_dot = Xsig_aug(4,i);
        double a = Xsig_aug(5,i);
        double phidd = Xsig_aug(6,i);
        double px_p , py_p;

        if(fabs(phi_dot) > 0.001){
            px_p = px + (v/phi_dot)*(sin(phi + phi_dot*delta_t) - sin(phi)) + (1/2)*pow(delta_t,2)*cos(phi)*a;
            py_p = py + (v/phi_dot)*(-cos(phi + phi_dot*delta_t) + cos(phi)) +  (1/2)*pow(delta_t,2)*sin(phi)*a;
        }
        else {
            px_p = px + v*cos(phi)*delta_t + (1/2)*pow(delta_t,2)*cos(phi)*a;
            py_p = py + v*sin(phi)*delta_t +  (1/2)*pow(delta_t,2)*sin(phi)*a;
        }

        double v_p , phi_p , phi_dot_p;

        v_p = v + delta_t*a;
        phi_p = phi + phi_dot*delta_t + (1/2)* pow(delta_t,2)*phidd;
        phi_dot_p = phi_dot + delta_t*phidd;

        VectorXd p_x(nx);
        p_x << px_p , py_p , v_p , phi_p , phi_dot_p;

        Xsig_pred.col(i) = p_x;
    }

    *Xsig_out = Xsig_pred;
}

void UKF::PredictMeanAndCovariance(VectorXd* x_pred,MatrixXd* P_pred){

    int nx = 5;
    int naug = 7;

    double lambda = 3 - naug;

    MatrixXd Xsig_pred(nx,2*naug+1);
    Xsig_pred << 5.9374,  6.0640,   5.925,  5.9436,  5.9266,  5.9374,  5.9389,  5.9374,  5.8106,  5.9457,  5.9310,  5.9465,  5.9374,  5.9359,  5.93744,
           1.48,  1.4436,   1.660,  1.4934,  1.5036,    1.48,  1.4868,    1.48,  1.5271,  1.3104,  1.4787,  1.4674,    1.48,  1.4851,    1.486,
          2.204,  2.2841,  2.2455,  2.2958,   2.204,   2.204,  2.2395,   2.204,  2.1256,  2.1642,  2.1139,   2.204,   2.204,  2.1702,   2.2049,
         0.5367, 0.47338, 0.67809, 0.55455, 0.64364, 0.54337,  0.5367, 0.53851, 0.60017, 0.39546, 0.51900, 0.42991, 0.530188,  0.5367, 0.535048,
          0.352, 0.29997, 0.46212, 0.37633,  0.4841, 0.41872,   0.352, 0.38744, 0.40562, 0.24347, 0.32926,  0.2214, 0.28687,   0.352, 0.318159;

    //vector for weights
    VectorXd weights(2*naug+1);  

    VectorXd x(nx);
    MatrixXd P(nx,nx);

    weights(0) = lambda / (lambda + naug);

    for (int i=1;i<2*naug+1;i++)
        weights(i) = 0.5 / (lambda + naug);

    x.fill(0.0);
    P.fill(0.0);
    for (int i=0;i <2*naug + 1;i++)
        x += weights(i) * Xsig_pred.col(i);


    for (int i=0;i <2*naug + 1;i++){
        VectorXd x_diff = (Xsig_pred.col(i) - x);
        //angle normalization
        while (x_diff(3) > M_PI)
            x_diff(3) -= 2.*M_PI;
        while (x_diff(3) < -M_PI)
            x_diff(3) += 2.*M_PI;

        P += weights(i)*x_diff*x_diff.transpose();
    }

     //print result
    cout << "Predicted state = " << endl;
    cout << x << endl;
    cout << "Predicted covariance matrix = " << endl;
    cout << P << endl;

  //write result
    *x_pred = x;
    *P_pred = P;       

}

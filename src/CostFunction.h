/*
 * CostFunction.h
 *
 *  Created on: Jun 12, 2015
 *      Author: sam
 */

#ifndef COSTFUNCTION_H_
#define COSTFUNCTION_H_

#include "Eigen/Dense"

class CostFunction
{
  public:
    virtual ~CostFunction()
    {
    }

    virtual Eigen::VectorXd configure(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y) =0;
    virtual Eigen::VectorXd getGrad(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y) =0;
    virtual double getCost(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y) =0;
    virtual double accuracy(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y) =0;

    double getNumGrad(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y);
    double getNumGrad(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y, const int& numChecks);
};

#endif /* COSTFUNCTION_H_ */
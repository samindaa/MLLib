/*
 * SupervisedNeuralNetworkCostFunction.h
 *
 *  Created on: Jun 15, 2015
 *      Author: sam
 */

#ifndef SUPERVISEDNEURALNETWORKCOSTFUNCTION_H_
#define SUPERVISEDNEURALNETWORKCOSTFUNCTION_H_

#include <vector>
#include "CostFunction.h"
#include "SigmoidFunction.h"
#include "SupervisedNeuralNetworkLayer.h"
#include "OpenMpParallel.h"

class SupervisedNeuralNetworkCostFunction: public CostFunction
{
  private:
    Eigen::VectorXd topology;
    std::vector<SupervisedNeuralNetworkLayer*> layers;
    int numberOfParameters;
    double LAMBDA;

  public:
    SupervisedNeuralNetworkCostFunction(const Eigen::VectorXd& topology,
        const double& LAMBDA = 0.0f);
    ~SupervisedNeuralNetworkCostFunction();

    Eigen::VectorXd configure(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
    Eigen::VectorXd getGrad(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y);
    double getCost(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y);
    double accuracy(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y);

  private:
    void toTheta(Eigen::VectorXd& grad);
    void toGradient(Eigen::VectorXd& grad);
    void toWeights(const Eigen::VectorXd& theta);
    void forwardPass(const Eigen::MatrixXd& X);
};

#endif /* SUPERVISEDNEURALNETWORKCOSTFUNCTION_H_ */

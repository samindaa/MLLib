/*
 * SoftICACostFunction.h
 *
 *  Created on: Jul 13, 2015
 *      Author: sam
 */

#ifndef SOFTICACOSTFUNCTION_H_
#define SOFTICACOSTFUNCTION_H_

#include "CostFunction.h"
#include <random>
#include <functional>

class SoftICACostFunction: public CostFunction
{
  private:
    int numFeatures; // number of filter banks to learn
    double lambda; // sparsity cost
    double epsilon; // epsilon to use in square-sqrt nonlinearity
    Eigen::MatrixXd W;
    Eigen::MatrixXd GradW;
    Eigen::MatrixXd XNorm;

  public:
    SoftICACostFunction(const int& numFeatures, const double& lambda, const double& epsilon);
    ~SoftICACostFunction();

    Eigen::VectorXd configure(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
    double evaluate(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y, Eigen::VectorXd& grad);
    double accuracy(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y);

  private:
    static double sample(double)
    {
      static std::random_device rd;
      static std::mt19937 gen(rd());
      std::normal_distribution<> d(0.0f, 1.0f);
      return d(gen);
    }
};

#endif /* SOFTICACOSTFUNCTION_H_ */

/*
 * SoftmaxCostFunction.cpp
 *
 *  Created on: Jun 15, 2015
 *      Author: sam
 */

#include "SoftmaxCostFunction.h"
#include <iostream>

SoftmaxCostFunction::SoftmaxCostFunction(const double& LAMBDA) :
    softmax(new SoftmaxFunction), LAMBDA(LAMBDA)
{
}

SoftmaxCostFunction::~SoftmaxCostFunction()
{
  delete softmax;
}

Eigen::VectorXd SoftmaxCostFunction::configure(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
{
  const int numberOfParameters = X.cols() * Y.cols();
  std::cout << "numberOfParameters: " << numberOfParameters << std::endl;
  Eigen::VectorXd theta = (Eigen::VectorXd::Random(numberOfParameters, 1).array() + 1.0f) * 0.5f
      * 0.001f;
  return theta;
}

double SoftmaxCostFunction::evaluate(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y, Eigen::VectorXd& grad)
{
  Eigen::MatrixXd Mat = softmax->getFunc(getMat(theta, X, Y));

  Eigen::MatrixXd Grad = -(X.transpose() * (Y - Mat));
  grad = Eigen::Map<Eigen::VectorXd>(Grad.data(), Grad.cols() * Grad.rows()) + (theta * LAMBDA);

  return -((Y.array() * Mat.array().log()).sum()) + (theta.array().square().sum()) * LAMBDA * 0.5f;
}

Eigen::MatrixXd SoftmaxCostFunction::getMat(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y)
{
  Eigen::VectorXd theta_tmp = theta;
  Eigen::Map<Eigen::MatrixXd> Theta(theta_tmp.data(), X.cols(), Y.cols()); // reshape
  return X * Theta;
}

double SoftmaxCostFunction::accuracy(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y)
{
  /*Eigen::VectorXd theta_tmp = theta;
   Eigen::MatrixXd Theta(Eigen::Map<Eigen::MatrixXd>(theta_tmp.data(), X.cols(), Y.cols())); // reshape*/

  Eigen::MatrixXd Mat = getMat(theta, X, Y);
  Eigen::MatrixXf::Index maxIndex;
  int correct = 0;
  int incorrect = 0;
//  omp_set_num_threads(NUMBER_OF_OPM_THREADS);
#pragma omp parallel for private(maxIndex) reduction(+:correct) reduction(+:incorrect)
  for (int i = 0; i < X.rows(); ++i)
  {
    Mat.row(i).maxCoeff(&maxIndex);
    if (Y(i, maxIndex) == 1)
      ++correct;
    else
    {
      ++incorrect;
      std::cout << i << std::endl;
      std::cout << "pred: " << Mat.row(i) << std::endl;
      std::cout << "true: " << Y.row(i) << std::endl;
    }
  }

  std::cout << "incorrect: " << incorrect << " outof: " << X.rows() << std::endl;
  return double(correct) * 100.0f / X.rows();
}


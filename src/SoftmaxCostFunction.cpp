/*
 * SoftmaxCostFunction.cpp
 *
 *  Created on: Jun 15, 2015
 *      Author: sam
 */

#include "SoftmaxCostFunction.h"
#include <iostream>
#include <fstream>

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
  Eigen::MatrixXd Mat = //
      softmax->getFunc(X * Eigen::Map<const Eigen::MatrixXd>(theta.data(), X.cols(), Y.cols()));
  Eigen::MatrixXd Grad = -(X.transpose() * (Y - Mat));
  grad = Eigen::Map<Eigen::VectorXd>(Grad.data(), Grad.cols() * Grad.rows()) + (theta * LAMBDA);
  return -((Y.array() * Mat.array().log()).sum()) + (theta.array().square().sum()) * LAMBDA * 0.5f;
}

double SoftmaxCostFunction::accuracy(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y)
{

  //std::ofstream ofs("m_test.txt");
  //ofs << X.topRows<10>() << std::endl;

  Eigen::MatrixXd Mat = X * Eigen::Map<const Eigen::MatrixXd>(theta.data(), X.cols(), Y.cols());
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
#pragma omp critical
      {
        std::cout << i << std::endl;
        std::cout << "pred: " << Mat.row(i) << std::endl;
        std::cout << "true: " << Y.row(i) << std::endl;
        //std::ofstream ofs("m_test.txt");
        //ofs << X.row(i) << std::endl;
      }
    }
  }

  std::cout << "incorrect: " << incorrect << " outof: " << X.rows() << std::endl;
  return double(correct) * 100.0f / X.rows();
}


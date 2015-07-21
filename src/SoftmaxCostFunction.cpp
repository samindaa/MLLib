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

Vector_t SoftmaxCostFunction::configure(const Matrix_t& X, const Matrix_t& Y)
{
  const int numberOfParameters = X.cols() * Y.cols();
  std::cout << "numberOfParameters: " << numberOfParameters << std::endl;
  Vector_t theta = (Vector_t::Random(numberOfParameters, 1).array() + 1.0f) * 0.5f * 0.001f;
  return theta;
}

double SoftmaxCostFunction::evaluate(const Vector_t& theta, const Matrix_t& X, const Matrix_t& Y,
    Vector_t& grad)
{
  Matrix_t Mat = //
      softmax->getFunc(X * Eigen::Map<const Matrix_t>(theta.data(), X.cols(), Y.cols()));
  Matrix_t Grad = -(X.transpose() * (Y - Mat));
  grad = Eigen::Map<Vector_t>(Grad.data(), Grad.cols() * Grad.rows()) + (theta * LAMBDA);
  return -((Y.array() * Mat.array().log()).sum()) + (theta.array().square().sum()) * LAMBDA * 0.5f;
}

double SoftmaxCostFunction::accuracy(const Vector_t& theta, const Matrix_t& X, const Matrix_t& Y)
{

  //std::ofstream ofs("m_test.txt");
  //ofs << X.topRows<10>() << std::endl;

  Matrix_t Mat = X * Eigen::Map<const Matrix_t>(theta.data(), X.cols(), Y.cols());
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


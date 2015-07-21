/*
 * SoftICACostFunction.cpp
 *
 *  Created on: Jul 13, 2015
 *      Author: sam
 */

#include "SoftICACostFunction.h"
#include <iostream>
#include <fstream>

SoftICACostFunction::SoftICACostFunction(const int& numFeatures, const double& lambda,
    const double& epsilon) :
    numFeatures(numFeatures), lambda(lambda), epsilon(epsilon)
{
}

SoftICACostFunction::~SoftICACostFunction()
{
}

Vector_t SoftICACostFunction::configure(const Matrix_t& X, const Matrix_t& Y)
{
  W = Matrix_t::Zero(numFeatures, X.cols()).unaryExpr(std::ptr_fun(SoftICACostFunction::sample));
  W.array() *= 0.01f;
  W = W.cwiseQuotient(W.cwiseProduct(W).rowwise().sum().matrix().replicate(1, W.cols()));
  Eigen::Map<Vector_t> theta(W.data(), W.rows() * W.cols());
  return theta;
}

double SoftICACostFunction::evaluate(const Vector_t& theta, const Matrix_t& Xin, const Matrix_t& Y,
    Vector_t& grad)
{
  W = Eigen::Map<const Matrix_t>(theta.data(), W.rows(), W.cols());
  const Matrix_t Wold = W;
  // project weights to norm ball (prevents degenerate bases)
  const double alpha = 1.0f;
  const double normeps = 1e-5;
  Vector_t epssumsq = (W.array().square().matrix().rowwise().sum().array() + normeps).matrix();
  Vector_t l2rows = epssumsq.array().sqrt() * alpha;
  W = l2rows.array().inverse().matrix().asDiagonal() * W;

  Matrix_t X = Xin.transpose();
  Matrix_t WX = W * X;
  Matrix_t WTWXmX = W.transpose() * WX - X;
  Matrix_t Fx = (WX.array().square() + epsilon).sqrt().matrix();

  Matrix_t GradW1 = WX.cwiseQuotient(Fx) * Xin;
  Matrix_t GradW2 = (W * WTWXmX * Xin) + (WX * WTWXmX.transpose());
  GradW = (GradW1 * lambda) + GradW2;

  //unproject gradient
  Matrix_t NewGradW = l2rows.array().inverse().matrix().asDiagonal() * GradW
      - (GradW.cwiseProduct(Wold).rowwise().sum().cwiseQuotient(epssumsq).asDiagonal() * W);

  grad = Eigen::Map<Vector_t>(NewGradW.data(), NewGradW.rows() * NewGradW.cols());

  return (Fx.array().sum() * lambda) + (WTWXmX.array().square().sum() / 2.0f);
}

double SoftICACostFunction::accuracy(const Vector_t& theta, const Matrix_t& X, const Matrix_t& Y)
{
  Vector_t theta_tmp = theta;
  W = Eigen::Map<Matrix_t>(theta_tmp.data(), W.rows(), W.cols());

  // debug
  std::ofstream ofs("../W.txt");
  ofs << W << std::endl;
  return 0.0f;
}


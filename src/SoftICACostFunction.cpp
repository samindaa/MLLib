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

Eigen::VectorXd SoftICACostFunction::configure(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
{
  Eigen::VectorXd xNorm = (X.cwiseProduct(X).rowwise().sum().array() + 1e-8).sqrt();
  XNorm = xNorm.array().inverse().matrix().asDiagonal();
  W = Eigen::MatrixXd::Zero(numFeatures, X.cols()).unaryExpr(
      std::ptr_fun(SoftICACostFunction::sample));
  W.array() *= 0.01f;
  W = (W.cwiseProduct(W).rowwise().sum().array().inverse().matrix().asDiagonal()) * W;
  Eigen::Map<Eigen::VectorXd> theta(W.data(), W.rows() * W.cols());
  return theta;
}

double SoftICACostFunction::evaluate(const Eigen::VectorXd& theta, const Eigen::MatrixXd& Xin,
    const Eigen::MatrixXd& Y, Eigen::VectorXd& grad)
{
  W = Eigen::Map<const Eigen::MatrixXd>(theta.data(), W.rows(), W.cols());
  const Eigen::MatrixXd Wold = W;
  // project weights to norm ball (prevents degenerate bases)
  const double alpha = 1.0f;
  const double normeps = 1e-5;
  Eigen::VectorXd epssumsq =
      (W.array().square().matrix().rowwise().sum().array() + normeps).matrix();
  Eigen::VectorXd l2rows = epssumsq.array().sqrt() * alpha;
  W = l2rows.array().inverse().matrix().asDiagonal() * W;

  Eigen::MatrixXd X = (XNorm * Xin).transpose();
  Eigen::MatrixXd WX = W * X;
  Eigen::MatrixXd WTWXmX = W.transpose() * WX - X;
  const double fx = sqrt(WX.array().square().sum() + epsilon);
  GradW = (WX * WTWXmX.transpose() + W * WTWXmX * X.transpose())
      + lambda * (WX * X.transpose()) / fx;

  //unproject gradient
  Eigen::MatrixXd NewGradW = l2rows.array().inverse().matrix().asDiagonal() * GradW
      - (GradW.cwiseProduct(Wold).rowwise().sum().cwiseQuotient(epssumsq).asDiagonal() * W);

  grad = Eigen::Map<Eigen::VectorXd>(NewGradW.data(), NewGradW.rows() * NewGradW.cols());
  //grad = Eigen::Map<Eigen::VectorXd>(GradW.data(), GradW.rows() * GradW.cols());
  return lambda * fx + WTWXmX.array().square().sum() * 0.5f;
}

double SoftICACostFunction::accuracy(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y)
{
  Eigen::VectorXd theta_tmp = theta;
  W = Eigen::Map<Eigen::MatrixXd>(theta_tmp.data(), W.rows(), W.cols());

  // debug
  std::ofstream ofs("../W.txt");
  ofs << W << std::endl;
  return 0.0f;
}


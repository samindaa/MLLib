//============================================================================
// Name        : stanford_dl_ex.cpp
// Author      : Sam Abeyruwan
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "Driver.h"
#include "HousingDataFunction.h"
#include "LIBLBFGSOptimizer.h"
#include "CppNumericalSolversOptimizer.h"
#include "LinearCostFunction.h"
#include "MNISTDataBinaryDigitsFunction.h"
#include "MNISTDataFunction.h"
#include "LogisticCostFunction.h"
#include "SoftmaxCostFunction.h"
#include "SupervisedNeuralNetworkCostFunction.h"
//
#include "IEEEHumanDataFunction.h"

void testMNISTDataFunction()
{
  MNISTDataFunction mnistdf;
  ConfigurationDescription cd;
  cd.config.insert(
      std::make_pair(mnistdf.trainImagesKey,
          "/home/sam/School/online/stanford_dl_ex/common/train-images-idx3-ubyte"));
  cd.config.insert(
      std::make_pair(mnistdf.trainLabelsKey,
          "/home/sam/School/online/stanford_dl_ex/common/train-labels-idx1-ubyte"));
  cd.config.insert(
      std::make_pair(mnistdf.testImagesKey,
          "/home/sam/School/online/stanford_dl_ex/common/t10k-images-idx3-ubyte"));
  cd.config.insert(
      std::make_pair(mnistdf.testLabelsKey,
          "/home/sam/School/online/stanford_dl_ex/common/t10k-labels-idx1-ubyte"));
  mnistdf.configure(&cd);
}

void testHousingDriver()
{
  ConfigurationDescription cd;
  cd.config["housing.data"] =
      "/home/sam/School/online/stanford_dl_ex/stanford_dl_ex/ex1/housing.data";
  HousingDataFunction hdf;
  LinearCostFunction hcf;
  LIBLBFGSOptimizer lbfgs;
//  CppNumericalSolversOptimizer lbfgs;
  Driver drv(&cd, &hdf, &hcf, &lbfgs);
  drv.drive();
}

void testMNISTBinaryDigitsDriver()
{
  MNISTDataBinaryDigitsFunction mnistdf;
  LogisticCostFunction mnistcf;
  ConfigurationDescription cd;
  cd.config.insert(
      std::make_pair(mnistdf.trainImagesKey,
          "/home/sam/School/online/stanford_dl_ex/common/train-images-idx3-ubyte"));
  cd.config.insert(
      std::make_pair(mnistdf.trainLabelsKey,
          "/home/sam/School/online/stanford_dl_ex/common/train-labels-idx1-ubyte"));
  cd.config.insert(
      std::make_pair(mnistdf.testImagesKey,
          "/home/sam/School/online/stanford_dl_ex/common/t10k-images-idx3-ubyte"));
  cd.config.insert(
      std::make_pair(mnistdf.testLabelsKey,
          "/home/sam/School/online/stanford_dl_ex/common/t10k-labels-idx1-ubyte"));

  LIBLBFGSOptimizer lbfgs;
//  CppNumericalSolversOptimizer lbfgs;
  Driver drv(&cd, &mnistdf, &mnistcf, &lbfgs);
  drv.drive();
}

void testMNISTDriver()
{
  MNISTDataFunction mnistdf;
  SoftmaxCostFunction mnistcf;
  ConfigurationDescription cd;
  cd.config.insert(
      std::make_pair(mnistdf.trainImagesKey,
          "/home/sam/School/online/stanford_dl_ex/common/train-images-idx3-ubyte"));
  cd.config.insert(
      std::make_pair(mnistdf.trainLabelsKey,
          "/home/sam/School/online/stanford_dl_ex/common/train-labels-idx1-ubyte"));
  cd.config.insert(
      std::make_pair(mnistdf.testImagesKey,
          "/home/sam/School/online/stanford_dl_ex/common/t10k-images-idx3-ubyte"));
  cd.config.insert(
      std::make_pair(mnistdf.testLabelsKey,
          "/home/sam/School/online/stanford_dl_ex/common/t10k-labels-idx1-ubyte"));

  LIBLBFGSOptimizer lbfgs;
//  CppNumericalSolversOptimizer lbfgs;
  Driver drv(&cd, &mnistdf, &mnistcf, &lbfgs);
  drv.drive();
}

void testEigenMap()
{
  Eigen::VectorXd theta;
  theta.setZero(4 * 5);
  for (int i = 0; i < theta.size(); ++i)
    theta(i) = i;

  std::cout << theta << std::endl;

  Eigen::Map<Eigen::MatrixXd> Theta(theta.data(), 4, 5); // reshape

  std::cout << Theta << std::endl;

  Eigen::VectorXd theta2(Eigen::Map<Eigen::VectorXd>(Theta.data(), 5 * 4));

  std::cout << theta2 << std::endl;

}

void testSupervisedNeuralNetworkCostFunction()
{
  Eigen::MatrixXd X = Eigen::MatrixXd::Random(5, 3);
  Eigen::MatrixXd Y = Eigen::MatrixXd::Identity(5, 4);
  Y(4, 3) = 1.0f; //<< fill in the missing I

  std::cout << X << std::endl;
  std::cout << Y << std::endl;

  Eigen::VectorXd topology(2);
  topology << 2, 6;

  SupervisedNeuralNetworkCostFunction nn(topology);
  Eigen::VectorXd theta = nn.configure(X, Y);
  double error = nn.getNumGrad(theta, X, Y, 20);
  std::cout << "error: " << error << std::endl;
}

void testMNISTSupervisedNeuralNetworkDriver()
{
  Eigen::VectorXd topology(2);
  topology << 20, 20;

  SupervisedNeuralNetworkCostFunction mnistcf(topology);

  MNISTDataFunction mnistdf;
  ConfigurationDescription cd;
  cd.config.insert(
      std::make_pair(mnistdf.trainImagesKey,
          "/home/sam/School/online/stanford_dl_ex/common/train-images-idx3-ubyte"));
  cd.config.insert(
      std::make_pair(mnistdf.trainLabelsKey,
          "/home/sam/School/online/stanford_dl_ex/common/train-labels-idx1-ubyte"));
  cd.config.insert(
      std::make_pair(mnistdf.testImagesKey,
          "/home/sam/School/online/stanford_dl_ex/common/t10k-images-idx3-ubyte"));
  cd.config.insert(
      std::make_pair(mnistdf.testLabelsKey,
          "/home/sam/School/online/stanford_dl_ex/common/t10k-labels-idx1-ubyte"));
  cd.addBiasTerm = false;
  LIBLBFGSOptimizer lbfgs;
//  CppNumericalSolversOptimizer lbfgs;
  Driver drv(&cd, &mnistdf, &mnistcf, &lbfgs);
  drv.drive();
}

void testIEEEHumanDataFunction()
{
  ConfigurationDescription cd;
  cd.config.insert(std::make_pair("exp", "__L1__"));
  IEEEHumanDataFunction ieeeHumanData;

  CostFunction* cf = nullptr;

  const bool isSoftmax = false;
  if (isSoftmax)
    cf = new SoftmaxCostFunction(0.01f);
  else
  {
    Eigen::VectorXd topology(1);
    topology << 5;
    cf = new SupervisedNeuralNetworkCostFunction(topology, 0.01f);
  }

  LIBLBFGSOptimizer lbfgs;
  //  CppNumericalSolversOptimizer lbfgs;
  Driver drv(&cd, &ieeeHumanData, cf, &lbfgs);
  drv.drive();

  delete cf;

}

int main()
{
  std::cout << "*** start ***" << std::endl;
//  testHousingDriver();
//  testMNISTDataFunction();
//  testMNISTBinaryDigitsDriver();
//  testMNISTDriver();
//  testEigenMap();
//  testSupervisedNeuralNetworkCostFunction();
  testIEEEHumanDataFunction();
//  testMNISTSupervisedNeuralNetworkDriver();
  std::cout << "*** end-- ***" << std::endl;
  return 0;
}

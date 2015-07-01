//============================================================================
// Name        : stanford_dl_ex.cpp
// Author      : Sam Abeyruwan
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <sstream>
#include <fstream>
#include "Driver.h"
#include "HousingDataFunction.h"
#include "LIBLBFGSOptimizer.h"
#include "LinearCostFunction.h"
#include "MNISTDataBinaryDigitsFunction.h"
#include "MNISTDataFunction.h"
#include "LogisticCostFunction.h"
#include "SoftmaxCostFunction.h"
#include "SupervisedNeuralNetworkCostFunction.h"
#include "DualToneMultiFrequencySignalingDataFunction.h"
#include "ConvolutionFunction.h"
#include "MeanPoolFunction.h"
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
  Driver drv(&cd, &mnistdf, &mnistcf, &lbfgs);
  drv.drive();
}

void testMNISTBinaryDigitsSupervisedNeuralNetworkCostFunctionDriver()
{
  Eigen::VectorXd topology(1);
  topology << 5;
  SupervisedNeuralNetworkCostFunction mnistcf(topology);
  //SoftmaxCostFunction mnistcf;
  //LogisticCostFunction mnistcf;
  MNISTDataBinaryDigitsFunction mnistdf(true);
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
  Eigen::VectorXd topology(1);
  topology << 100;

  SupervisedNeuralNetworkCostFunction mnistcf(topology, 0.01f);

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
  Driver drv(&cd, &mnistdf, &mnistcf, &lbfgs);
  drv.drive();
}

void testIEEEHumanDataFunction()
{
  ConfigurationDescription cd;
  const int h = 2;
  switch (h)
  {
    case 0:
      cd.config.insert(std::make_pair("exp", "__L1__"));
      break;
    case 1:
      cd.config.insert(std::make_pair("exp", "__L2_true__"));
      break;
    case 2:
      cd.config.insert(std::make_pair("exp", "__L2_negg__"));
  }

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
  Driver drv(&cd, &ieeeHumanData, cf, &lbfgs);
  drv.drive();

  delete cf;

}

void testDualToneMultiFrequencySignalingDataFunction()
{
  DualToneMultiFrequencySignalingDataFunction h;
  h.configure(nullptr);
}

// For testing
class RandomFilterFunction: public FilterFunction
{
  private:
    int filterDim;
    int numFilters;

  public:
    RandomFilterFunction(const int& filterDim, const int& numFilters) :
        filterDim(filterDim), numFilters(numFilters)
    {
    }

    void configure()
    {
      config << filterDim, filterDim;
      Weights.setRandom(filterDim * filterDim, numFilters);
      biases.setRandom(numFilters);
      //Weights.setOnes(filterDim * filterDim, numFilters);
      //biases.setOnes(numFilters);
    }
};

void testConvolutionAndPool()
{
  if (true)
  {
    const int filterDim = 8;
    const int imageDim = 28;
    const int poolDim = 3;
    const int numFilters = 100;
    const int outputDim = (imageDim - filterDim + 1) / poolDim;
    RandomFilterFunction rff(filterDim, numFilters);
    rff.configure();

    SigmoidFunction sf;
    ConvolutionFunction cf(&rff, &sf);
    MeanPoolFunction pf(numFilters, outputDim);
    //MeanPoolFunction mpf;
    Eigen::Vector2i config;
    config << imageDim, imageDim;

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
    cd.meanStddNormalize = false;

    mnistdf.configure(&cd);

    ConvolutedFunctions* convolvedFeatures = nullptr;
    for (int i = 0; i < 13; ++i)
      convolvedFeatures = cf.conv(mnistdf.getTrainingX().topRows<199>(), config);

    assert(convolvedFeatures->convolutions.size() == 199);
    for (auto i = convolvedFeatures->convolutions.begin();
        i != convolvedFeatures->convolutions.end(); ++i)
      assert((int )i->second.size() == rff.getWeights().cols());

    // Validate convoluations
    Eigen::MatrixXd ConvImages = mnistdf.getTrainingX().topRows<8>();
    for (int i = 0; i < 1000; ++i)
    {
      const int filterNum = rand() % rff.getWeights().cols();
      const int imageNum = rand() % 8;
      const int imageRow = rand() % (config(0) - rff.getConfig()(0) + 1);
      const int imageCol = rand() % (config(1) - rff.getConfig()(1) + 1);

      Eigen::VectorXd im = ConvImages.row(imageNum);
      Eigen::Map<Eigen::MatrixXd> Image(im.data(), config(0), config(1));

      Eigen::MatrixXd Patch = Image.block(imageRow, imageCol, rff.getConfig()(0),
          rff.getConfig()(1));

      // Filter
      Eigen::Map<Eigen::MatrixXd> W(rff.getWeights().col(filterNum).data(), rff.getConfig()(0),
          rff.getConfig()(1));
      const double b = rff.getBiases()(filterNum);

      double feature = Patch.cwiseProduct(W).sum() + b;
      feature = 1.0f / (1.0f + exp(-feature));

      if (fabs(
          feature - convolvedFeatures->convolutions[imageNum][filterNum]->X(imageRow, imageCol))
          > 1e-9)
      {
        std::cout << "Convolved feature does not match test feature: " << i << std::endl;
        std::cout << "Filter Number: " << filterNum << std::endl;
        std::cout << "Image Number: " << imageNum << std::endl;
        std::cout << "Image Row: " << imageRow << std::endl;
        std::cout << "Image Col: " << imageCol << std::endl;
        std::cout << "Convolved feature: "
            << convolvedFeatures->convolutions[imageNum][filterNum]->X(imageRow, imageCol)
            << std::endl;
        std::cout << "Test feature: " << feature << std::endl;
        std::cout << "Convolved feature does not match test feature" << std::endl;
        exit(EXIT_FAILURE);
      }

    }

    // Pool
    PooledFunctions* pooling = nullptr;
    for (int i = 0; i < 13; ++i)
      pooling = pf.pool(convolvedFeatures, poolDim);

    assert((int )pooling->pooling.size() == 199);
    for (auto iter = pooling->pooling.begin(); iter != pooling->pooling.end(); ++iter)
    {
      assert(iter->second.size() == (size_t )rff.getWeights().cols());
      for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2)
      {
        assert(iter2->second->X.rows() == (config(0) - rff.getConfig()(0) + 1) / 3);
        assert(iter2->second->X.rows() == 7);
        assert(iter2->second->X.cols() == (config(0) - rff.getConfig()(0) + 1) / 3);
        assert(iter2->second->X.cols() == 7);
      }
    }

  }

  if (true)
  {
    // test pool function

    Eigen::VectorXd testVec(64);
    for (int i = 0; i < testVec.size(); ++i)
      testVec(i) = i + 1;
    Eigen::Map<Eigen::MatrixXd> TestMatrix(testVec.data(), 8, 8);

    std::cout << "TestMatrix: " << std::endl;
    std::cout << TestMatrix << std::endl;

    Eigen::MatrixXd ExpectedMatrix(2, 2);
    ExpectedMatrix(0, 0) = TestMatrix.block(0, 0, 4, 4).array().mean();
    ExpectedMatrix(0, 1) = TestMatrix.block(0, 4, 4, 4).array().mean();
    ExpectedMatrix(1, 0) = TestMatrix.block(4, 0, 4, 4).array().mean();
    ExpectedMatrix(1, 1) = TestMatrix.block(4, 4, 4, 4).array().mean();

    std::cout << "Expected: " << std::endl;
    std::cout << ExpectedMatrix << std::endl;

    ConvolutedFunctions cfs;
    ConvolutedFunction xcf;
    xcf.X = TestMatrix;
    cfs.convolutions[0].insert(std::make_pair(0, (&xcf)));

    MeanPoolFunction testMpf(1, 2);
    PooledFunctions* pfs = testMpf.pool(&cfs, 4);

    assert(pfs->pooling.size() == 1);
    assert(pfs->pooling[0].size() == 1);
    Eigen::MatrixXd PX = pfs->pooling[0][0]->X;

    std::cout << "Obtain: " << std::endl;
    std::cout << PX << std::endl;
  }

}

int main()
{
  std::cout << "*** start ***" << std::endl;
//  testHousingDriver();
//  testMNISTDataFunction();
//  testMNISTBinaryDigitsDriver();
//  testMNISTBinaryDigitsSupervisedNeuralNetworkCostFunctionDriver();
//  testMNISTDriver();
//  testEigenMap();
//  testSupervisedNeuralNetworkCostFunction();
//  testIEEEHumanDataFunction();
//  testMNISTSupervisedNeuralNetworkDriver();
//  testDualToneMultiFrequencySignalingDataFunction();
  testConvolutionAndPool();
  std::cout << "*** end-- ***" << std::endl;
  return 0;
}

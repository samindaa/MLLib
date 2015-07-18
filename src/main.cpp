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
#include <random>
#include <functional>
#include <thread>
//
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
#include "ConvolutionalNeuralNetworkCostFunction.h"
#include "SGDOptimizer.h"
#include "WhiteningFunction.h"
#include "MNISTSamplePatchesDataFunction.h"
#include "SoftICACostFunction.h"
#include "NaturalImageDataFunction.h"
#include "StopWatch.h"
//
#include "IEEEHumanDataFunction.h"

void updateMNISTConfig(Config& config)
{
  config.setValue(trainImagesKeyMnist,
      std::string("/home/sam/School/online/stanford_dl_ex/common/train-images-idx3-ubyte"));
  config.setValue(trainLabelsKeyMnist,
      std::string("/home/sam/School/online/stanford_dl_ex/common/train-labels-idx1-ubyte"));
  config.setValue(testImagesKeyMnist,
      std::string("/home/sam/School/online/stanford_dl_ex/common/t10k-images-idx3-ubyte"));
  config.setValue(testLabelsKeyMnist,
      std::string("/home/sam/School/online/stanford_dl_ex/common/t10k-labels-idx1-ubyte"));
}

void testMNISTDataFunction()
{
  MNISTDataFunction mnistdf;
  Config config;
  updateMNISTConfig(config);
  mnistdf.configure(&config);
}

void testHousingDriver()
{
  Config config;
  config.setValue("housing.data",
      std::string("/home/sam/School/online/stanford_dl_ex/stanford_dl_ex/ex1/housing.data"));
  HousingDataFunction hdf;
  LinearCostFunction hcf;
  LIBLBFGSOptimizer lbfgs;
  Driver drv(&config, &hdf, &hcf, &lbfgs);
  drv.drive();
}

void testMNISTBinaryDigitsDriver()
{
  MNISTDataBinaryDigitsFunction mnistdf;
  LogisticCostFunction mnistcf;
  Config config;
  updateMNISTConfig(config);

  LIBLBFGSOptimizer lbfgs;
  Driver drv(&config, &mnistdf, &mnistcf, &lbfgs);
  drv.drive();
}

void testSoftmaxCostFunction()
{
  MNISTDataBinaryDigitsFunction mnistdf(true);
  Config config;
  updateMNISTConfig(config);
  mnistdf.configure(&config);

  SoftmaxCostFunction mnistcf;
  Eigen::VectorXd theta = mnistcf.configure(mnistdf.getTrainingX(), mnistdf.getTrainingY());

  mnistcf.getNumGrad(theta, mnistdf.getTrainingX(), mnistdf.getTrainingY(), 5);
}

void testSoftmaxCostFunctionDriver()
{
  MNISTDataBinaryDigitsFunction mnistdf(true);
  Config config;
  updateMNISTConfig(config);
  SoftmaxCostFunction mnistcf(1.0f);
  LIBLBFGSOptimizer lbfgs;
  //config.setValue("debugMode", true);
  config.setValue("numGrd", true);
  //config.setValue("addBiasTerm", false);
  Driver drv(&config, &mnistdf, &mnistcf, &lbfgs);
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
  Config config;
  updateMNISTConfig(config);
  config.setValue("addBiasTerm", false);

  LIBLBFGSOptimizer lbfgs;
  Driver drv(&config, &mnistdf, &mnistcf, &lbfgs);
  drv.drive();
}

void testMNISTDriver()
{
  MNISTDataFunction mnistdf;
  SoftmaxCostFunction mnistcf;
  Config config;
  updateMNISTConfig(config);
  LIBLBFGSOptimizer lbfgs;
//  CppNumericalSolversOptimizer lbfgs;
  Driver drv(&config, &mnistdf, &mnistcf, &lbfgs);
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
  Config config;
  updateMNISTConfig(config);
  config.setValue("addBiasTerm", false);
  LIBLBFGSOptimizer lbfgs;
  Driver drv(&config, &mnistdf, &mnistcf, &lbfgs);
  drv.drive();
}

void testIEEEHumanDataFunction()
{
  Config config;
  const int h = 2;
  switch (h)
  {
    case 0:
      config.setValue("exp", std::string("__L1__"));
      break;
    case 1:
      config.setValue("exp", std::string("__L2_true__"));
      break;
    case 2:
      config.setValue("exp", std::string("__L2_negg__"));
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
  Driver drv(&config, &ieeeHumanData, cf, &lbfgs);
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
    Eigen::Vector2i configImageDim;
    configImageDim << imageDim, imageDim;

    MNISTDataFunction mnistdf;
    Config config;
    updateMNISTConfig(config);
    config.setValue("addBiasTerm", false);
    config.setValue("meanStddNormalize", false);

    mnistdf.configure(&config);

    Convolutions* convolvedFeatures = nullptr;
    for (int i = 0; i < 13; ++i)
      convolvedFeatures = cf.conv(mnistdf.getTrainingX().topRows<199>(), configImageDim);

    assert(convolvedFeatures->unordered_map.size() == 199);
    for (auto i = convolvedFeatures->unordered_map.begin();
        i != convolvedFeatures->unordered_map.end(); ++i)
      assert((int )i->second.size() == rff.getWeights().cols());

    // Validate convoluations
    Eigen::MatrixXd ConvImages = mnistdf.getTrainingX().topRows<8>();
    for (int i = 0; i < 1000; ++i)
    {
      const int filterNum = rand() % rff.getWeights().cols();
      const int imageNum = rand() % 8;
      const int imageRow = rand() % (configImageDim(0) - rff.getConfig()(0) + 1);
      const int imageCol = rand() % (configImageDim(1) - rff.getConfig()(1) + 1);

      Eigen::VectorXd im = ConvImages.row(imageNum);
      Eigen::Map<Eigen::MatrixXd> Image(im.data(), configImageDim(0), configImageDim(1));

      Eigen::MatrixXd Patch = Image.block(imageRow, imageCol, rff.getConfig()(0),
          rff.getConfig()(1));

      // Filter
      Eigen::Map<Eigen::MatrixXd> W(rff.getWeights().col(filterNum).data(), rff.getConfig()(0),
          rff.getConfig()(1));
      const double b = rff.getBiases()(filterNum);

      double feature = Patch.cwiseProduct(W).sum() + b;
      feature = 1.0f / (1.0f + exp(-feature));

      if (fabs(
          feature - convolvedFeatures->unordered_map[imageNum][filterNum]->X(imageRow, imageCol))
          > 1e-9)
      {
        std::cout << "Convolved feature does not match test feature: " << i << std::endl;
        std::cout << "Filter Number: " << filterNum << std::endl;
        std::cout << "Image Number: " << imageNum << std::endl;
        std::cout << "Image Row: " << imageRow << std::endl;
        std::cout << "Image Col: " << imageCol << std::endl;
        std::cout << "Convolved feature: "
            << convolvedFeatures->unordered_map[imageNum][filterNum]->X(imageRow, imageCol)
            << std::endl;
        std::cout << "Test feature: " << feature << std::endl;
        std::cout << "Convolved feature does not match test feature" << std::endl;
        exit(EXIT_FAILURE);
      }

    }

    // Pool
    Poolings* pooling = nullptr;
    for (int i = 0; i < 13; ++i)
      pooling = pf.pool(convolvedFeatures, poolDim);

    assert((int )pooling->unordered_map.size() == 199);
    for (auto iter = pooling->unordered_map.begin(); iter != pooling->unordered_map.end(); ++iter)
    {
      assert(iter->second.size() == (size_t )rff.getWeights().cols());
      for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2)
      {
        assert(iter2->second->X.rows() == (configImageDim(0) - rff.getConfig()(0) + 1) / 3);
        assert(iter2->second->X.rows() == 7);
        assert(iter2->second->X.cols() == (configImageDim(0) - rff.getConfig()(0) + 1) / 3);
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

    Convolutions cfs;
    Convolution xcf;
    xcf.X = TestMatrix;
    cfs.unordered_map[0].insert(std::make_pair(0, (&xcf)));

    MeanPoolFunction testMpf(1, 2);
    Poolings* pfs = testMpf.pool(&cfs, 4);

    assert(pfs->unordered_map.size() == 1);
    assert(pfs->unordered_map[0].size() == 1);
    Eigen::MatrixXd PX = pfs->unordered_map[0][0]->X;

    std::cout << "Obtain: " << std::endl;
    std::cout << PX << std::endl;
  }

}

void testConvolutionalNeuralNetworkCostFunction()
{
  MNISTDataFunction mnistdf;
  Config config;
  updateMNISTConfig(config);
  config.setValue("addBiasTerm", false);
  config.setValue("meanStddNormalize", false);
  config.setValue("debugMode", true);
  mnistdf.configure(&config);

  const int imageDim = 28; // height/width of image
  const int filterDim = 9; // dimension of convolutional filter
  const int numFilters = 2; // number of convolutional filters
  const int poolDim = 5; // dimension of pooling area
  const int numClasses = 10; // number of classes to predict

  ConvolutionalNeuralNetworkCostFunction cnn(imageDim, filterDim, numFilters, poolDim, numClasses);
  Eigen::VectorXd theta = cnn.configure(mnistdf.getTrainingX(), mnistdf.getTrainingY());

  std::cout << "theta: " << theta.size() << std::endl;
  Eigen::VectorXd grad;
  double cost = cnn.evaluate(theta, mnistdf.getTrainingX(), mnistdf.getTrainingY(), grad);

  std::cout << "cost: " << cost << std::endl;
  std::cout << "grad: " << grad.size() << std::endl;

  double error = cnn.getNumGrad(theta, mnistdf.getTrainingX(), mnistdf.getTrainingY());
  std::cout << "error: " << error << std::endl;
}

void testKroneckorTensorProduct()
{
  Eigen::MatrixXd A(2, 2);
  A << 1, 2, 3, 4;
  std::cout << A << std::endl;

  Eigen::MatrixXd Ones = Eigen::MatrixXd::Ones(2, 2);
  std::cout << A << std::endl;

  Eigen::MatrixXd C(A.rows() * Ones.rows(), A.cols() * Ones.cols());
  Eigen::KroneckerProduct<Eigen::MatrixXd, Eigen::MatrixXd> Kron(A, Ones);
  Kron.evalTo(C);

  std::cout << C << std::endl;
}

void testMNISTConvolutionalNeuralNetworkDriver()
{
  MNISTDataFunction mnistdf;
  Config config;
  updateMNISTConfig(config);
  config.setValue("addBiasTerm", false);
  config.setValue("meanStddNormalize", false);
  SGDOptimizer sgd;

  const int imageDim = 28; // height/width of image
  const int filterDim = 9; // dimension of convolutional filter
  const int numFilters = 20; // number of convolutional filters
  const int poolDim = 2; // dimension of pooling area
  const int numClasses = 10; // number of classes to predict

  ConvolutionalNeuralNetworkCostFunction cf(imageDim, filterDim, numFilters, poolDim, numClasses);
  Driver drv(&config, &mnistdf, &cf, &sgd);
  drv.drive();
}

double sample(double)
{
  static std::random_device rd;
  static std::mt19937 gen(rd());

  // values near the mean are the most likely
  // standard deviation affects the dispersion of generated values from the mean
  std::normal_distribution<> d(0.0f, 1.0f);
  return d(gen);
}

void testNormalRandomWithEigen()
{
  Eigen::MatrixXd m = Eigen::MatrixXd::Zero(3, 3).unaryExpr(std::ptr_fun(sample));
  std::cout << m << std::endl;
}

void testEigenVectorizedOperations()
{
  const Eigen::MatrixXd m0 = Eigen::MatrixXd::Zero(3, 3).unaryExpr(std::ptr_fun(sample));
  std::cout << "m: " << std::endl;
  std::cout << m0 << std::endl;

  const Eigen::MatrixXd m1 = m0.cwiseProduct(m0);
  std::cout << "m0: " << std::endl;
  std::cout << m0 << std::endl;
  std::cout << "m1: " << std::endl;
  std::cout << m1 << std::endl;

  const Eigen::MatrixXd m2 = m0.array().square().matrix();
  std::cout << "m0: " << std::endl;
  std::cout << m0 << std::endl;
  std::cout << "m2: " << std::endl;
  std::cout << m2 << std::endl;

}

void testWhiteningFunction()
{
  MNISTDataFunction mnistdf;
  Config config;

  config.setValue("debugMode", false);
  config.setValue("addBiasTerm", false);
  config.setValue("meanStddNormalize", false);
  config.setValue("configurePolicyTesting", false);
  config.setValue("zeroMean", true);
  config.setValue("pcaWhitening", true);
  config.setValue("zcaWhitening", true);

  updateMNISTConfig(config);
  mnistdf.configure(&config);

  WhiteningFunction wf(&config);
  Eigen::MatrixXd XZCAWhite = wf.gen(mnistdf.getTrainingX());

  // debug
  std::ofstream xofs("X100.txt");
  std::ofstream zofs("Z100.txt");

  xofs << mnistdf.getTrainingX().topRows<100>() << std::endl;
  zofs << XZCAWhite.topRows<100>() << std::endl;
}

void testMNISTSamplePatchesDataFunction()
{
  const int numPatches = 20; // 10000
  const int patchWidth = 9;

  MNISTSamplePatchesDataFunction mnistdf(numPatches, patchWidth);
  Config config;

  config.setValue("addBiasTerm", false);
  config.setValue("meanStddNormalize", false);
  config.setValue("configurePolicyTesting", false);
  config.setValue("trainingMeanAndStdd", false);

  updateMNISTConfig(config);
  mnistdf.configure(&config);

  if (true)
  {
    //debug
    std::cout << mnistdf.getTrainingX().rows() << " x " << mnistdf.getTrainingX().cols()
        << std::endl;
    //std::ofstream of("../patches.txt");
    //of << mnistdf.getTrainingX() << std::endl;
  }
}

void testSoftICACostFunction()
{
  const int numPatches = 500; // 10000
  const int patchWidth = 9;
  MNISTSamplePatchesDataFunction mnistdf(numPatches, patchWidth);
  Config config;

  config.setValue("debugMode", true);
  config.setValue("addBiasTerm", false);
  config.setValue("meanStddNormalize", false);
  config.setValue("configurePolicyTesting", false);
  config.setValue("trainingMeanAndStdd", false);

  updateMNISTConfig(config);
  mnistdf.configure(&config);

  const int numFeatures = 5; // 50
  const double lambda = 0.0005f;
  const double epsilon = 1e-2;

  SoftICACostFunction sf(numFeatures, lambda, epsilon);

  Eigen::VectorXd theta = sf.configure(mnistdf.getTrainingX(), mnistdf.getTrainingY());

  std::cout << "theta: " << theta.size() << std::endl;
  Eigen::VectorXd grad;
  double cost = sf.evaluate(theta, mnistdf.getTrainingX(), mnistdf.getTrainingY(), grad);

  std::cout << "cost: " << cost << std::endl;
  std::cout << "grad: " << grad.size() << std::endl;

  double error = sf.getNumGrad(theta, mnistdf.getTrainingX(), mnistdf.getTrainingY());
  std::cout << "error: " << error << std::endl;

}

void testSoftICADriver()
{
  const int numPatches = 10000;
  const int patchWidth = 9;
  MNISTSamplePatchesDataFunction mnistdf(numPatches, patchWidth);
  Config config;
  config.setValue("addBiasTerm", false);
  config.setValue("meanStddNormalize", false);
  config.setValue("configurePolicyTesting", false);
  config.setValue("trainingMeanAndStdd", false);
  updateMNISTConfig(config);

  const int numFeatures = 50;
  const double lambda = 0.0005f;
  const double epsilon = 1e-2;

  SoftICACostFunction sfc(numFeatures, lambda, epsilon);

  LIBLBFGSOptimizer lbfgs;
  Driver drv(&config, &mnistdf, &sfc, &lbfgs);
  drv.drive();

}

void testNaturalImageDataFunction()
{

  const int numPatches = 50; // 10000
  const int patchWidth = 20;

  NaturalImageDataFunction nidf(numPatches, patchWidth);
  Config config;
  nidf.configure(&config);
}

void testSoftICADriver2()
{
  const int numPatches = 20000;
  const int patchWidth = 8;
  NaturalImageDataFunction nidf(numPatches, patchWidth);
  Config config;

  const int numFeatures = 50;
  const double lambda = 0.0005f;
  const double epsilon = 1e-2;

  SoftICACostFunction sfc(numFeatures, lambda, epsilon);

  LIBLBFGSOptimizer lbfgs;
  Driver drv(&config, &nidf, &sfc, &lbfgs);
  drv.drive();

}

void testStopWatch()
{
  StopWatch stopWatch;
  stopWatch.start();
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  stopWatch.end();

  std::cout << "ms: " << stopWatch.ms_count() << " us: " << stopWatch.us_count() << std::endl;
}

void testEigenConstMap()
{
  Eigen::VectorXd x(4);
  x << 1, 2, 3, 4;
  std::cout << x << std::endl;

  const Eigen::VectorXd y = x;

  std::cout << y << std::endl;

  Eigen::MatrixXd X = Eigen::Map<const Eigen::MatrixXd>(y.data(), 2, 2);

  std::cout << X << std::endl;
}

int main()
{
  std::cout << "*** start ***" << std::endl;
//  testHousingDriver();
//  testMNISTDataFunction();
//  testMNISTBinaryDigitsDriver();
//  testSoftmaxCostFunction();
  testSoftmaxCostFunctionDriver();
//  testMNISTBinaryDigitsSupervisedNeuralNetworkCostFunctionDriver();
//  testMNISTDriver();
//  testEigenMap();
//  testSupervisedNeuralNetworkCostFunction();
//  testIEEEHumanDataFunction();
//  testMNISTSupervisedNeuralNetworkDriver();
//  testDualToneMultiFrequencySignalingDataFunction();
//  testConvolutionAndPool();
//  testConvolutionalNeuralNetworkCostFunction();
//  testMNISTConvolutionalNeuralNetworkDriver();
//  testKroneckorTensorProduct();
//  testNormalRandomWithEigen();
//  testWhiteningFunction();
//  testMNISTSamplePatchesDataFunction();
//  testSoftICACostFunction();
//  testSoftICADriver();
//  testEigenVectorizedOperations();
//  testNaturalImageDataFunction();
//  testSoftICADriver2();
//  testStopWatch();
//  testEigenConstMap();
  std::cout << "*** end-- ***" << std::endl;
  return 0;
}

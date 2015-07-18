/*
 * ConvolutionalNeuralNetworkCostFunction.cpp
 *
 *  Created on: Jun 30, 2015
 *      Author: sam
 */

#include "ConvolutionalNeuralNetworkCostFunction.h"
#include <iostream>

ConvolutionalNeuralNetworkCostFunction::ConvolutionalNeuralNetworkCostFunction(const int& imageDim,
    const int& filterDim, const int& numFilters, const int& poolDim, const int& numClasses) :
    imageDim(imageDim), filterDim(filterDim), numFilters(numFilters), poolDim(poolDim), //
    numClasses(numClasses), convDim(imageDim - filterDim + 1), outputDim(convDim / poolDim), //
    cnnFilterFunction(new CNNFilterFunction(filterDim, numFilters)), //
    sigmoidActivationFunction(new SigmoidFunction()), //
    convolutionFunction(new ConvolutionFunction(cnnFilterFunction, sigmoidActivationFunction)), //
    poolingFunction(new MeanPoolFunction(numFilters, outputDim)), //
    softmaxActivationFunction(new SoftmaxFunction()), numberOfParameters(0), LAMBDA(0.0f)
{
  // fixme: size checking
}

ConvolutionalNeuralNetworkCostFunction::~ConvolutionalNeuralNetworkCostFunction()
{
  delete cnnFilterFunction;
  delete sigmoidActivationFunction;
  delete convolutionFunction;
  delete poolingFunction;
  delete softmaxActivationFunction;
}

Eigen::VectorXd ConvolutionalNeuralNetworkCostFunction::configure(const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y)
{
  assert(filterDim < imageDim);

  config << imageDim, imageDim;

  cnnFilterFunction->configure(); // Wc, bc

  assert(convDim == 20);

  assert(convDim % poolDim == 0);

  const int hiddenSize = std::pow(outputDim, 2) * numFilters;

  const double r = sqrt(6.0f) / sqrt(numClasses + hiddenSize + 1.0f);

  Wd = Eigen::MatrixXd::Random(hiddenSize, numClasses) * r;
  bd.setZero(numClasses);

  WdGrad.setZero(hiddenSize, numClasses);
  bdGrad.setZero(numClasses);

  //LAMBDA = 3e-3;

  numberOfParameters = cnnFilterFunction->getWeights().size()
      + cnnFilterFunction->getBiases().size() + Wd.size() + bd.size();

  std::cout << "numberOfParameters: " << numberOfParameters << std::endl;

  Eigen::VectorXd theta(numberOfParameters);

  toTheta(theta);

  return theta;

}

void ConvolutionalNeuralNetworkCostFunction::toTheta(Eigen::VectorXd& theta)
{
  int j = 0;

  // Wc
  Eigen::VectorXd thetaLayerWc(Eigen::Map<Eigen::VectorXd>(cnnFilterFunction->getWeights().data(), //
      cnnFilterFunction->getWeights().rows() * cnnFilterFunction->getWeights().cols()));
  theta.segment(j, thetaLayerWc.size()) = thetaLayerWc;
  j += thetaLayerWc.size();
  // Wb
  theta.segment(j, cnnFilterFunction->getBiases().size()) = cnnFilterFunction->getBiases();
  j += cnnFilterFunction->getBiases().size();

  // Wd
  Eigen::VectorXd thetaLayerWd(Eigen::Map<Eigen::VectorXd>(Wd.data(), Wd.rows() * Wd.cols()));
  theta.segment(j, thetaLayerWd.size()) = thetaLayerWd;
  j += thetaLayerWd.size();
  // bd
  theta.segment(j, bd.size()) = bd;
  j += bd.size();

  assert(j == numberOfParameters);

}

void ConvolutionalNeuralNetworkCostFunction::toGrad(Eigen::VectorXd& grad)
{
  int j = 0;

  // Wc
  Eigen::VectorXd thetaLayerWc(
      Eigen::Map<Eigen::VectorXd>(cnnFilterFunction->getWeightsGrad().data(),
          cnnFilterFunction->getWeightsGrad().rows() * cnnFilterFunction->getWeightsGrad().cols()));
  grad.segment(j, thetaLayerWc.size()) = thetaLayerWc;
  j += thetaLayerWc.size();
  // Wb
  grad.segment(j, cnnFilterFunction->getBiasesGrad().size()) = cnnFilterFunction->getBiasesGrad();
  j += cnnFilterFunction->getBiasesGrad().size();

  // Wd
  Eigen::VectorXd thetaLayerWd(
      Eigen::Map<Eigen::VectorXd>(WdGrad.data(), WdGrad.rows() * WdGrad.cols()));
  grad.segment(j, thetaLayerWd.size()) = thetaLayerWd;
  j += thetaLayerWd.size();
  // bd
  grad.segment(j, bdGrad.size()) = bdGrad;
  j += bdGrad.size();

  assert(j == numberOfParameters);
}

void ConvolutionalNeuralNetworkCostFunction::toWeights(const Eigen::VectorXd& theta)
{
  Eigen::VectorXd theta_tmp = theta;
  int j = 0;

  // Wc
  cnnFilterFunction->getWeights() = Eigen::Map<Eigen::MatrixXd>(theta_tmp.data() + j,
      cnnFilterFunction->getWeights().rows(), cnnFilterFunction->getWeights().cols());
  j += cnnFilterFunction->getWeights().size();
  // bc
  cnnFilterFunction->getBiases() = Eigen::Map<Eigen::VectorXd>(theta_tmp.data() + j,
      cnnFilterFunction->getBiases().size());
  j += cnnFilterFunction->getBiases().size();

  // Wd
  Wd = Eigen::Map<Eigen::MatrixXd>(theta_tmp.data() + j, Wd.rows(), Wd.cols());
  j += Wd.size();
  // bd
  bd = Eigen::Map<Eigen::VectorXd>(theta_tmp.data() + j, bd.size());
  j += bd.size();

  assert(j == numberOfParameters);
}

double ConvolutionalNeuralNetworkCostFunction::evaluate(const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, Eigen::VectorXd& grad)
{
  /* STEP 1a: Forward Propagation
   In this step you will forward propagate the input through the
   convolutional and subsampling (mean pooling) layers.  You will then use
   the responses from the convolution and pooling layer as the input to a
   standard softmax layer.
   */

  /* Convolutional Layer
   For each image and each filter, convolve the image with the filter, add
   the bias and apply the sigmoid nonlinearity.  Then subsample the
   convolved activations with mean pooling.  Store the results of the
   convolution in activations and the results of the pooling in
   activationsPooled.  You will need to save the convolved activations for
   backpropagation.
   */
  toWeights(theta);

  Convolutions* activations = convolutionFunction->conv(X, config);

  const bool DEBUG = false;

  if (DEBUG)
  {
    std::cout << "activations: " << std::endl;
    //debug
    for (auto iter = activations->unordered_map.begin(); iter != activations->unordered_map.end();
        ++iter)
    {
      std::cout << "iter: " << iter->first << std::endl;
      for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2)
      {
        std::cout << "\t iter2: " << iter2->first << " => " << iter2->second->X.rows() << " x "
            << iter2->second->X.cols() << std::endl;
        //std::cout << iter2->second->X << std::endl;

        std::cout << "^^^" << std::endl;
      }
    }
  }

  Poolings* activationsPooled = poolingFunction->pool(activations, poolDim);

  if (DEBUG)
  {
    std::cout << "activationsPooled: " << std::endl;
    //debug
    for (auto iter = activationsPooled->unordered_map.begin();
        iter != activationsPooled->unordered_map.end(); ++iter)
    {
      std::cout << "iter: " << iter->first << std::endl;
      for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2)
      {
        std::cout << "\t iter2: " << iter2->first << std::endl;
        std::cout << iter2->second->X << std::endl;
        std::cout << "^^^" << std::endl;
      }
    }

  }
  /*
   Reshape activations into 2-d matrix, hiddenSize x numImages,
   for Softmax layer
   */

  ActivationsPooled = Eigen::MatrixXd::Zero(activationsPooled->unordered_map.size(),
      std::pow(outputDim, 2) * numFilters); // fixme:

  for (auto iter = activationsPooled->unordered_map.begin();
      iter != activationsPooled->unordered_map.end(); ++iter)
    //ActivationsPooled.row(iter->first) = activationsPooled->toVector(iter->first).transpose();
    activationsPooled->toMatrix(ActivationsPooled, iter->first);

  if (DEBUG)
  {
    std::cout << "ActivationsPooled: " << ActivationsPooled.rows() << " x "
        << ActivationsPooled.cols() << std::endl;
    std::cout << ActivationsPooled << std::endl;
  }
  /*std::cout << "From: " << std::endl;
   activationsPooled->toPoolingDelta(ActivationsPooled);

   //debug
   for (auto iter = activationsPooled->poolingDelta.begin();
   iter != activationsPooled->poolingDelta.end(); ++iter)
   {
   std::cout << "iter: " << iter->first << std::endl;
   for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2)
   {
   std::cout << "\t iter2: " << iter2->first << std::endl;
   std::cout << iter2->second->X << std::endl;
   std::cout << "^^^" << std::endl;
   }
   }
   */
  /* Softmax Layer
   Forward propagate the pooled activations calculated above into a
   standard softmax layer. For your convenience we have reshaped
   activationPooled into a hiddenSize x numImages matrix.  Store the
   results in probs.

   numClasses x numImages for storing probability that each image belongs to
   each class.
   */

  OutputA = ((ActivationsPooled * Wd).rowwise() + bd.transpose()).matrix();
  OutputZ = softmaxActivationFunction->getFunc(OutputA);

  /* STEP 1c: Backpropagation
   Backpropagate errors through the softmax and convolutional/subsampling
   layers.  Store the errors for the next step to calculate the gradient.
   Backpropagating the error w.r.t the softmax layer is as usual.  To
   backpropagate through the pooling layer, you will need to upsample the
   error with respect to the pooling layer for each filter and each image.
   Use the kron function and a matrix of ones to do this upsampling
   quickly.
   */

  /* STEP 1d: Gradient Calculation
   After backpropagating the errors above, we can use them to calculate the
   gradient with respect to all the parameters.  The gradient w.r.t the
   softmax layer is calculated as usual.  To calculate the gradient w.r.t.
   a filter in the convolutional layer, convolve the backpropagated error
   for that filter with each image and aggregate over images.
   */

  // Dense Layer
  OutputDelta = (OutputZ.array() - Y.array()).matrix();
  WdGrad = ActivationsPooled.transpose() * OutputDelta;
  bdGrad = OutputDelta.colwise().sum().transpose();

  // Convoluation Layer
  if (DEBUG)
  {
    std::cout << "\n\n Convoluation Layer: " << std::endl;
  }

  PoolingDelta = OutputDelta * Wd.transpose();
  activationsPooled->toPoolingSensitivities(PoolingDelta);

  if (DEBUG)
  {
    std::cout << PoolingDelta << std::endl;
  }

  if (DEBUG)
  {
    std::cout << "PoolingDelta: " << std::endl;
    for (auto iter = activationsPooled->unordered_map_sensitivities.begin();
        iter != activationsPooled->unordered_map_sensitivities.end(); ++iter)
    {
      std::cout << "iter: " << iter->first << std::endl;
      for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2)
      {
        std::cout << "\t iter2: " << iter2->first << std::endl;
        std::cout << iter2->second->X << std::endl;
        std::cout << "^^^" << std::endl;
      }
    }
  }

  if (DEBUG)
  {
    std::cout << "delta_pool: " << std::endl;
  }

  poolingFunction->delta_pool(poolDim);

  if (DEBUG)
  {
    std::cout << "upsample: " << std::endl;
  }

  for (auto iter = activationsPooled->unordered_map_sensitivities.begin();
      iter != activationsPooled->unordered_map_sensitivities.end(); ++iter)
  {
    if (DEBUG)
    {
      std::cout << "iter: " << iter->first << std::endl;
    }

    for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2)
    {

      Eigen::MatrixXd UnPool = iter2->second->X;
      Eigen::MatrixXd Z = activations->unordered_map[iter->first][iter2->first]->X;
      iter2->second->X = sigmoidActivationFunction->getGrad(Z).cwiseProduct(UnPool);

      if (DEBUG)
      {
        std::cout << "\t iter2: " << iter2->first << " => " << iter2->second->X.rows() << " x "
            << iter2->second->X.cols() << std::endl;

        //std::cout << iter2->second->X << std::endl;
        std::cout << "^^^" << std::endl;
      }
    }
  }

  if (DEBUG)
  {
    std::cout << "getBiasesGrad: " << "getWeightsGrad: " << std::endl;
  }

  cnnFilterFunction->getBiasesGrad().setZero(cnnFilterFunction->getBiasesGrad().size());
  cnnFilterFunction->getWeightsGrad().setZero(cnnFilterFunction->getWeightsGrad().rows(),
      cnnFilterFunction->getWeightsGrad().cols());
  Eigen::MatrixXd Conv;
  // Each image
  for (int i = 0; i < X.rows(); ++i)
  {
    Eigen::VectorXd x = X.row(i);
    Eigen::Map<Eigen::MatrixXd> I(x.data(), config(0), config(1));

    auto& iter = activationsPooled->unordered_map_sensitivities[i];

    for (auto iter2 = iter.begin(); iter2 != iter.end(); ++iter2)
    {
      Eigen::MatrixXd& W = iter2->second->X;

      cnnFilterFunction->getBiasesGrad()(iter2->first) += W.sum();

      if (DEBUG)
      {
        std::cout << "iter2: " << iter2->first << std::endl;
        std::cout << "I: " << I.rows() << " x " << I.cols() << std::endl;
        std::cout << "W: " << W.rows() << " x " << W.cols() << std::endl;
      }

      convolutionFunction->validConv(Conv, I, W, 0.0f);

      if (DEBUG)
      {
        std::cout << "conv2: " << Conv.rows() << " x " << Conv.cols() << std::endl;
      }

      Eigen::Map<Eigen::VectorXd> tmp(Conv.data(), Conv.size());
      cnnFilterFunction->getWeightsGrad().col(iter2->first) += tmp;
    }
  }

  grad.resize(numberOfParameters);
  toGrad(grad);
  grad.array() += (theta.array() * LAMBDA);

  return -((Y.array() * OutputZ.array().log()).sum())
      + (theta.array().square().sum()) * LAMBDA * 0.5f;;
}

double ConvolutionalNeuralNetworkCostFunction::accuracy(const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& X0, const Eigen::MatrixXd& Y)
{
  toWeights(theta);

  int correct = 0;
  int incorrect = 0;

  convolutionFunction->clear();
  poolingFunction->clear();

  for (int i = 0; i < X0.rows(); ++i)
  {
    Eigen::MatrixXd X = X0.row(i);
    Convolutions* activations = convolutionFunction->conv(X, config);
    Poolings* activationsPooled = poolingFunction->pool(activations, poolDim);
    ActivationsPooled = Eigen::MatrixXd::Zero(activationsPooled->unordered_map.size(),
        std::pow(outputDim, 2) * numFilters); // fixme:
    for (auto iter = activationsPooled->unordered_map.begin();
        iter != activationsPooled->unordered_map.end(); ++iter)
      //ActivationsPooled.row(iter->first) = activationsPooled->toVector(iter->first).transpose();
      activationsPooled->toMatrix(ActivationsPooled, iter->first);
    OutputA = ((ActivationsPooled * Wd).rowwise() + bd.transpose()).matrix();
    OutputZ = softmaxActivationFunction->getFunc(OutputA);

    Eigen::MatrixXf::Index maxIndex;

    OutputZ.row(0).maxCoeff(&maxIndex);
    if (Y(i, maxIndex) == 1)
      ++correct;
    else
    {
      ++incorrect;
      //std::cout << i << std::endl;
      //std::cout << "pred: " << OutputZ.row(i) << std::endl;
      //std::cout << "true: " << Y.row(i) << std::endl;
    }
  }
  std::cout << "incorrect: " << incorrect << " outof: " << X0.rows() << std::endl;
  return double(correct) * 100.0f / X0.rows();
}


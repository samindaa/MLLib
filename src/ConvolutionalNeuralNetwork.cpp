/*
 * ConvolutionalNeuralNetwork.cpp
 *
 *  Created on: Jun 30, 2015
 *      Author: sam
 */

#include "ConvolutionalNeuralNetwork.h"

ConvolutionalNeuralNetwork::ConvolutionalNeuralNetwork() :
    imageDim(28), filterDim(9), numFilters(20), poolDim(2), numClasses(10), //
    convDim(imageDim - filterDim + 1), outputDim(convDim / poolDim), //
    filterFunction(new CNNFilterFunction(filterDim, numFilters)), //
    activationFunction(new SigmoidFunction()), //
    convolutionFunction(new ConvolutionFunction(filterFunction, activationFunction)), //
    poolFunction(new MeanPoolFunction(numFilters, outputDim)), //
    softmaxActivationFunction(new SoftmaxFunction()), numberOfParameters(0)
{
  // fixme: size checking
}

ConvolutionalNeuralNetwork::~ConvolutionalNeuralNetwork()
{
  delete filterFunction;
  delete activationFunction;
  delete convolutionFunction;
  delete poolFunction;
  delete softmaxActivationFunction;
}

Eigen::VectorXd ConvolutionalNeuralNetwork::configure(const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y)
{
  assert(filterDim < imageDim);

  config << imageDim, imageDim;

  filterFunction->configure(); // Wc, bc

  const int dimensionOfConvolvedImage = imageDim - filterDim + 1;

  assert(dimensionOfConvolvedImage == 20);

  assert(dimensionOfConvolvedImage % poolDim == 0);

  const int dimensionOfPoolingImage = dimensionOfConvolvedImage / poolDim;

  const int hiddenSize = std::pow(dimensionOfPoolingImage, 2) * numFilters;

  const double r = sqrt(6.0f) / sqrt(numClasses + hiddenSize + 1.0f);

  Wd = Eigen::MatrixXd::Random(hiddenSize, numClasses) * r;
  bd.setZero(numClasses);

  WdGrad.setZero(hiddenSize, numClasses);
  bdGrad.setZero(numClasses);

  numberOfParameters = filterFunction->getWeights().size() + filterFunction->getBiases().size() //
      + Wd.size() + bd.size();

  Eigen::VectorXd theta(numberOfParameters);

  toTheta(theta);

  return theta;

}

void ConvolutionalNeuralNetwork::toTheta(Eigen::VectorXd& theta)
{
  int j = 0;

  // Wc
  Eigen::VectorXd thetaLayerWc(
      Eigen::Map<Eigen::VectorXd>(filterFunction->getWeights().data(),
          filterFunction->getWeights().rows() * filterFunction->getWeights().cols()));
  theta.segment(j, thetaLayerWc.size()) = thetaLayerWc;
  j += thetaLayerWc.size();
  // Wb
  theta.segment(j, filterFunction->getBiases().size()) = filterFunction->getBiases();
  j += filterFunction->getBiases().size();

  // Wd
  Eigen::VectorXd thetaLayerWd(Eigen::Map<Eigen::VectorXd>(Wd.data(), Wd.rows() * Wd.cols()));
  theta.segment(j, thetaLayerWd.size()) = thetaLayerWd;
  j += thetaLayerWd.size();
  // bd
  theta.segment(j, bd.size()) = bd;
  j += bd.size();

  assert(j == numberOfParameters);

}

void ConvolutionalNeuralNetwork::toWeights(const Eigen::VectorXd& theta)
{
  Eigen::VectorXd theta_tmp = theta;
  int j = 0;

  // Wc
  filterFunction->getWeights() = Eigen::Map<Eigen::MatrixXd>(theta_tmp.data() + j,
      filterFunction->getWeights().rows(), filterFunction->getWeights().cols());
  j += filterFunction->getWeights().size();
  // bc
  filterFunction->getBiases() = Eigen::Map<Eigen::VectorXd>(theta_tmp.data() + j,
      filterFunction->getBiases().size());
  j += filterFunction->getBiases().size();

  // Wd
  Wd = Eigen::Map<Eigen::MatrixXd>(theta_tmp.data() + j, Wd.rows(), Wd.cols());
  j += Wd.size();
  // bd
  bd = Eigen::Map<Eigen::VectorXd>(theta_tmp.data() + j, bd.size());
  j += bd.size();

  assert(j == numberOfParameters);
}

double ConvolutionalNeuralNetwork::evaluate(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y, Eigen::VectorXd& grad)
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

  ConvolutedFunctions* activations = convolutionFunction->conv(X, config);
  PooledFunctions* activationsPooled = poolFunction->pool(activations, poolDim);

  /*
   Reshape activations into 2-d matrix, hiddenSize x numImages,
   for Softmax layer
   */

  ActivationsPooled = Eigen::MatrixXd::Zero(activationsPooled->pooling.size(),
      std::pow(outputDim, 2) * numFilters); // fixme:

  for (auto iter = activationsPooled->pooling.begin(); iter != activationsPooled->pooling.end();
      ++iter)
    ActivationsPooled.row(iter->first) = activationsPooled->flatten(iter->first).transpose();

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

  return 0.0f;
}

double ConvolutionalNeuralNetwork::accuracy(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y)
{
  return 0.0f;
}


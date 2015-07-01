/*
 * SupervisedNeuralNetworkCostFunction.cpp
 *
 *  Created on: Jun 15, 2015
 *      Author: sam
 */

#include "SupervisedNeuralNetworkCostFunction.h"
#include <iostream>

SupervisedNeuralNetworkCostFunction::SupervisedNeuralNetworkCostFunction(
    const Eigen::VectorXd& topology, const double& LAMBDA) :
    topology(topology), numberOfParameters(0), LAMBDA(LAMBDA)
{
}

SupervisedNeuralNetworkCostFunction::~SupervisedNeuralNetworkCostFunction()
{
  for (std::vector<SupervisedNeuralNetworkLayer*>::iterator iter = layers.begin();
      iter != layers.end(); ++iter)
    delete *iter;
  layers.clear();
}

Eigen::VectorXd SupervisedNeuralNetworkCostFunction::configure(const Eigen::MatrixXd& X,
    const Eigen::MatrixXd& Y)
{
  SupervisedNeuralNetworkLayer* firstLayer = new SupervisedNeuralNetworkLayer(X.cols(),
      SupervisedNeuralNetworkLayer::NULL_FUNCTION);
  SupervisedNeuralNetworkLayer* finalLayer = new SupervisedNeuralNetworkLayer(Y.cols(),
      SupervisedNeuralNetworkLayer::SOFTMAX_FUNCTION);

  layers.push_back(firstLayer);
  for (int i = 0; i < topology.size(); ++i)
  {
    SupervisedNeuralNetworkLayer* hiddenLayer = new SupervisedNeuralNetworkLayer(topology(i),
        SupervisedNeuralNetworkLayer::SIGMOD_FUNCTION);
    hiddenLayer->W.setRandom(layers[i]->size, hiddenLayer->size);
    hiddenLayer->b.setRandom(hiddenLayer->size);
    hiddenLayer->W *= 0.001f;
    hiddenLayer->b *= 0.001f;
    layers.push_back(hiddenLayer);
    numberOfParameters += hiddenLayer->W.size();
    numberOfParameters += hiddenLayer->b.size();
  }
  finalLayer->W = Eigen::MatrixXd::Random(layers[layers.size() - 1]->size, finalLayer->size);
  finalLayer->b = Eigen::VectorXd::Random(finalLayer->size);
  finalLayer->W *= 0.001f;
  finalLayer->b *= 0.001f;
  layers.push_back(finalLayer);
  numberOfParameters += finalLayer->W.size();
  numberOfParameters += finalLayer->b.size();

  std::cout << "n0: " << layers[0]->size << std::endl;
  for (size_t i = 1; i < layers.size(); ++i)
  {
    std::cout << "n" << i << ": " << layers[i]->size << std::endl;
    std::cout << "\t W: " << layers[i]->W.rows() << " x " << layers[i]->W.cols() << std::endl;
    std::cout << "\t b: " << layers[i]->b.size() << std::endl;
  }

  std::cout << "numberOfParameters: " << numberOfParameters << std::endl;

  Eigen::VectorXd theta(numberOfParameters);
  toTheta(theta);

  return theta;
}

void SupervisedNeuralNetworkCostFunction::toTheta(Eigen::VectorXd& theta)
{
  int j = 0;
  for (size_t i = 1; i < layers.size(); ++i)
  {
    Eigen::VectorXd thetaLayer(
        Eigen::Map<Eigen::VectorXd>(layers[i]->W.data(),
            layers[i]->W.cols() * layers[i]->W.rows()));

    theta.segment(j, thetaLayer.size()) = thetaLayer;
    j += thetaLayer.size();
    theta.segment(j, layers[i]->b.size()) = layers[i]->b;
    j += layers[i]->b.size();
  }
  assert(j == numberOfParameters);
}

void SupervisedNeuralNetworkCostFunction::toGradient(Eigen::VectorXd& grad)
{
  int j = 0;
  for (size_t i = 1; i < layers.size(); ++i)
  {
    Eigen::VectorXd thetaLayer(
        Eigen::Map<Eigen::VectorXd>(layers[i]->GradientW.data(),
            layers[i]->GradientW.cols() * layers[i]->GradientW.rows()));

    grad.segment(j, thetaLayer.size()) = thetaLayer;
    j += thetaLayer.size();
    grad.segment(j, layers[i]->gradientb.size()) = layers[i]->gradientb;
    j += layers[i]->gradientb.size();
  }
  assert(j == numberOfParameters);

}

void SupervisedNeuralNetworkCostFunction::toWeights(const Eigen::VectorXd& theta)
{
  Eigen::VectorXd theta_tmp = theta;
  int j = 0;
  for (size_t i = 1; i < layers.size(); ++i)
  {
    layers[i]->W = Eigen::Map<Eigen::MatrixXd>(theta_tmp.data() + j, layers[i]->W.rows(),
        layers[i]->W.cols());
    j += layers[i]->W.size();
    layers[i]->b = Eigen::Map<Eigen::VectorXd>(theta_tmp.data() + j, layers[i]->b.size());
    j += layers[i]->b.size();
  }
  assert(j == numberOfParameters);
}

void SupervisedNeuralNetworkCostFunction::forwardPass(const Eigen::MatrixXd& X)
{
  // Input layer
  layers[0]->Z = X;
  //Hidden layers to output layer
  for (size_t j = 1; j < layers.size(); ++j)
  {
    layers[j]->A =
        ((layers[j - 1]->Z * layers[j]->W).rowwise() + layers[j]->b.transpose()).matrix();
    layers[j]->Z = layers[j]->function->getFunc(layers[j]->A);
  }

}

double SupervisedNeuralNetworkCostFunction::evaluate(const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y, Eigen::VectorXd& grad)
{
  toWeights(theta);
  forwardPass(X);

  // for classification
  size_t j = layers.size() - 1;
  layers[j]->Delta = (layers[j]->Z.array() - Y.array()).matrix();

  layers[j]->GradientW = layers[j - 1]->Z.transpose() * layers[j]->Delta;
  layers[j]->gradientb = layers[j]->Delta.colwise().sum().transpose();
  --j;
  for (; j > 0; --j)
  {
    Eigen::MatrixXd DeltaWT = layers[j + 1]->Delta * layers[j + 1]->W.transpose();
    layers[j]->Delta = layers[j]->function->getGrad(layers[j]->Z).cwiseProduct(DeltaWT);

    layers[j]->GradientW = layers[j - 1]->Z.transpose() * layers[j]->Delta;
    layers[j]->gradientb = layers[j]->Delta.colwise().sum().transpose();
  }

  grad.resize(numberOfParameters);
  toGradient(grad);
  grad.array() += (theta.array() * LAMBDA);

  return -((Y.array() * layers[layers.size() - 1]->Z.array().log()).sum())
      + (theta.array().square().sum()) * LAMBDA * 0.5f;

}

double SupervisedNeuralNetworkCostFunction::accuracy(const Eigen::VectorXd& theta,
    const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y)
{
  toWeights(theta);
  forwardPass(X);

  Eigen::MatrixXf::Index maxIndex;
  int correct = 0;
  int incorrect = 0;
  //omp_set_num_threads(NUMBER_OF_OPM_THREADS);
//#pragma omp parallel for private(maxIndex) reduction(+:correct) reduction(+:incorrect)
  for (int i = 0; i < X.rows(); ++i)
  {
    layers[layers.size() - 1]->Z.row(i).maxCoeff(&maxIndex);
    if (Y(i, maxIndex) == 1)
      ++correct;
    else
    {
      ++incorrect;
      std::cout << i << std::endl;
      std::cout << "pred: " << layers[layers.size() - 1]->Z.row(i) << std::endl;
      std::cout << "true: " << Y.row(i) << std::endl;
    }
  }
  std::cout << "incorrect: " << incorrect << " outof: " << X.rows() << std::endl;
  return double(correct) * 100.0f / X.rows();

}


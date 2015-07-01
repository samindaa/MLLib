/*
 * ConvolutionalNeuralNetwork.h
 *
 *  Created on: Jun 30, 2015
 *      Author: sam
 */

#ifndef CONVOLUTIONALNEURALNETWORK_H_
#define CONVOLUTIONALNEURALNETWORK_H_

#include "CostFunction.h"
#include "FilterFunction.h"
#include "SigmoidFunction.h"
#include "ConvolutionFunction.h"
#include "MeanPoolFunction.h"
#include "SoftmaxFunction.h"

class ConvolutionalNeuralNetwork: public CostFunction
{
  private:
    class CNNFilterFunction: public FilterFunction
    {
      private:
        int filterDim;
        int numFilters;

      public:
        CNNFilterFunction(const int& filterDim, const int& numFilters) :
            filterDim(filterDim), numFilters(numFilters)
        {
        }

        void configure()
        {
          config << filterDim, filterDim;
          Weights.setRandom(filterDim * filterDim, numFilters);
          Weights.array() *= 1e-1;
          biases.setZero(numFilters);

          WeightsGrad.setZero(filterDim * filterDim, numFilters);
          biases.setZero(numFilters);
        }
    };

    int imageDim; // height/width of image
    int filterDim; // dimension of convolutional filter
    int numFilters; // number of convolutional filters
    int poolDim; // dimension of pooling area
    int numClasses; // number of classes to predict
    int convDim; // dimension of convolved output
    int outputDim; // dimension of subsampled output

    Eigen::Vector2i config;

    CNNFilterFunction* filterFunction;
    ActivationFunction* activationFunction;
    ConvolutionFunction* convolutionFunction;
    PoolFunction* poolFunction;
    ActivationFunction* softmaxActivationFunction;

    Eigen::MatrixXd Wd;
    Eigen::VectorXd bd;

    Eigen::MatrixXd WdGrad;
    Eigen::VectorXd bdGrad;

    Eigen::MatrixXd ActivationsPooled;

    Eigen::MatrixXd OutputA;
    Eigen::MatrixXd OutputZ;

    int numberOfParameters;

  public:
    ConvolutionalNeuralNetwork();
    ~ConvolutionalNeuralNetwork();

    Eigen::VectorXd configure(const Eigen::MatrixXd& X, const Eigen::MatrixXd& Y);
    double evaluate(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y, Eigen::VectorXd& grad);
    double accuracy(const Eigen::VectorXd& theta, const Eigen::MatrixXd& X,
        const Eigen::MatrixXd& Y);

  private:
    void toTheta(Eigen::VectorXd& theta);
    void toWeights(const Eigen::VectorXd& theta);

};

#endif /* CONVOLUTIONALNEURALNETWORK_H_ */

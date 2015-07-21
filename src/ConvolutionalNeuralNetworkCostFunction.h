/*
 * ConvolutionalNeuralNetworkCostFunction.h
 *
 *  Created on: Jun 30, 2015
 *      Author: sam
 */

#ifndef CONVOLUTIONALNEURALNETWORKCOSTFUNCTION_H_
#define CONVOLUTIONALNEURALNETWORKCOSTFUNCTION_H_

#include <random>
#include <functional>
//
#include "CostFunction.h"
#include "FilterFunction.h"
#include "SigmoidFunction.h"
#include "ConvolutionFunction.h"
#include "MeanPoolFunction.h"
#include "SoftmaxFunction.h"

class ConvolutionalNeuralNetworkCostFunction: public CostFunction
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

        static double sample(double)
        {
          static std::random_device rd;
          static std::mt19937 gen(rd());
          std::normal_distribution<> d(0.0f, 1.0f);
          return d(gen);
        }

        void configure()
        {
          config << filterDim, filterDim;
          Weights = Matrix_t::Zero(filterDim * filterDim, numFilters).unaryExpr(
              std::ptr_fun(CNNFilterFunction::sample));
          Weights.array() *= 1e-1;
          biases.setZero(numFilters);

          WeightsGrad.setZero(filterDim * filterDim, numFilters);
          biasesGrad.setZero(numFilters);
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

    CNNFilterFunction* cnnFilterFunction;
    ActivationFunction* sigmoidActivationFunction;
    ConvolutionFunction* convolutionFunction;
    PoolingFunction* poolingFunction;
    ActivationFunction* softmaxActivationFunction;

    Matrix_t Wd;
    Vector_t bd;

    Matrix_t WdGrad;
    Vector_t bdGrad;

    Matrix_t ActivationsPooled;

    Matrix_t OutputA;
    Matrix_t OutputZ;

    Matrix_t OutputDelta;
    Matrix_t PoolingDelta;

    int numberOfParameters;
    double LAMBDA;

  public:
    ConvolutionalNeuralNetworkCostFunction(const int& imageDim, const int& filterDim,
        const int& numFilters, const int& poolDim, const int& numClasses);
    ~ConvolutionalNeuralNetworkCostFunction();

    Vector_t configure(const Matrix_t& X, const Matrix_t& Y);
    double evaluate(const Vector_t& theta, const Matrix_t& X, const Matrix_t& Y, Vector_t& grad);
    double accuracy(const Vector_t& theta, const Matrix_t& X, const Matrix_t& Y);

  private:
    void toTheta(Vector_t& theta);
    void toGrad(Vector_t& grad);
    void toWeights(const Vector_t& theta);

};

#endif /* CONVOLUTIONALNEURALNETWORKCOSTFUNCTION_H_ */

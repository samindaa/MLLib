/*
 * FilterFunction.h
 *
 *  Created on: Jun 27, 2015
 *      Author: sam
 */

#ifndef FILTERFUNCTION_H_
#define FILTERFUNCTION_H_

#include "EigenFunction.h"

class FilterFunction: public EigenFunction
{
  protected:
    Eigen::MatrixXd Weights;
    Eigen::VectorXd biases;
    Eigen::Vector2i config;

    Eigen::MatrixXd WeightsGrad;
    Eigen::VectorXd biasesGrad;

  public:
    virtual ~FilterFunction()
    {
    }

    virtual void configure() =0;

    Eigen::MatrixXd& getWeights()
    {
      return Weights;
    }

    Eigen::VectorXd& getBiases()
    {
      return biases;
    }

    Eigen::MatrixXd& getWeightsGrad()
    {
      return WeightsGrad;
    }

    Eigen::VectorXd& getBiasesGrad()
    {
      return biasesGrad;
    }

    Eigen::Vector2i& getConfig()
    {
      return config;
    }

};

#endif /* FILTERFUNCTION_H_ */

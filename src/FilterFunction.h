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
    Matrix_t Weights;
    Vector_t biases;
    Eigen::Vector2i config;

    Matrix_t WeightsGrad;
    Vector_t biasesGrad;

  public:
    virtual ~FilterFunction()
    {
    }

    virtual void configure() =0;

    Matrix_t& getWeights()
    {
      return Weights;
    }

    Vector_t& getBiases()
    {
      return biases;
    }

    Matrix_t& getWeightsGrad()
    {
      return WeightsGrad;
    }

    Vector_t& getBiasesGrad()
    {
      return biasesGrad;
    }

    Eigen::Vector2i& getConfig()
    {
      return config;
    }

};

#endif /* FILTERFUNCTION_H_ */

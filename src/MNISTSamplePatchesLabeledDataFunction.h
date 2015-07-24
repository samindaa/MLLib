/*
 * MNISTSamplePatchesLabeledDataFunction.h
 *
 *  Created on: Jul 21, 2015
 *      Author: sam
 */

#ifndef MNISTSAMPLEPATCHESLABELEDDATAFUNCTION_H_
#define MNISTSAMPLEPATCHESLABELEDDATAFUNCTION_H_

#include "MNISTDataFunction.h"
#include "ConvolutionFunction.h"
#include "PoolingFunction.h"

class MNISTSamplePatchesLabeledDataFunction: public MNISTDataFunction
{
  private:
    ConvolutionFunction* convolutionFunction;
    PoolingFunction* poolingFunction;
    Eigen::Vector2i convolutionFunctionConfig;
    int numFilters;
    int poolDim;
    int outputDim;

  public:
    MNISTSamplePatchesLabeledDataFunction(ConvolutionFunction* convolutionFunction,
        PoolingFunction* poolingFunction, const Eigen::Vector2i& convolutionFunctionConfig,
        const int& numFilters, const int& poolDim, const int& outputDim);
    void configurePolicy(const Matrix_t& tmpX, Matrix_t& X, const Matrix_t& tmpY, Matrix_t& Y);
    void update(const Matrix_t& tmpX, const Matrix_t& tmpY, Matrix_t& X, Matrix_t& Y,
        const int& index, const int& size);
};

#endif /* MNISTSAMPLEPATCHESLABELEDDATAFUNCTION_H_ */

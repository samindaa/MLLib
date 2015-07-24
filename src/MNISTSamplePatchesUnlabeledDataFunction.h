/*
 * MNISTSamplePatchesUnlabeledDataFunction.h
 *
 *  Created on: Jul 21, 2015
 *      Author: sam
 */

#ifndef MNISTSAMPLEPATCHESUNLABELEDDATAFUNCTION_H_
#define MNISTSAMPLEPATCHESUNLABELEDDATAFUNCTION_H_

#include "MNISTSamplePatchesDataFunction.h"

class MNISTSamplePatchesUnlabeledDataFunction: public MNISTSamplePatchesDataFunction
{
  public:
    MNISTSamplePatchesUnlabeledDataFunction(const int& numPatches, const int& patchWidth);
    ~MNISTSamplePatchesUnlabeledDataFunction();
    void configurePolicy(const Matrix_t& tmpX, Matrix_t& X, const Matrix_t& tmpY, Matrix_t& Y);
};

#endif /* MNISTSAMPLEPATCHESUNLABELEDDATAFUNCTION_H_ */

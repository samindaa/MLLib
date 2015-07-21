/*
 * MNISTSamplePatchesDataFunction.h
 *
 *  Created on: Jul 13, 2015
 *      Author: sam
 */

#include "MNISTDataFunction.h"
#include "WhiteningFunction.h"

class MNISTSamplePatchesDataFunction: public MNISTDataFunction
{
  private:
    int numPatches;
    int patchWidth;
    WhiteningFunction* whiteningFunction;
    Config config;

  public:
    MNISTSamplePatchesDataFunction(const int& numPatches, const int& patchWidth);
    ~MNISTSamplePatchesDataFunction();
    void configurePolicy(const Matrix_t& tmpX, Matrix_t& X, const Matrix_t& tmpY, Matrix_t& Y);
};


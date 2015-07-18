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
    void configurePolicy(const Eigen::MatrixXd& tmpX, Eigen::MatrixXd& X,
        const Eigen::MatrixXd& tmpY, Eigen::MatrixXd& Y);
};


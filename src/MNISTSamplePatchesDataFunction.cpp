/*
 * MNISTSamplePatchesDataFunction.cpp
 *
 *  Created on: Jul 13, 2015
 *      Author: sam
 */

#include "MNISTSamplePatchesDataFunction.h"

#include <iostream>
#include <random>

MNISTSamplePatchesDataFunction::MNISTSamplePatchesDataFunction(const int& numPatches,
    const int& patchWidth) :
    numPatches(numPatches), patchWidth(patchWidth)
{
  config.setValue("zeroMean", true);
  config.setValue("pcaWhitening", true);
  config.setValue("zcaWhitening", true);
  config.setValue("epsilon", 0.0f);

  whiteningFunction = new WhiteningFunction(&config);
}

MNISTSamplePatchesDataFunction::~MNISTSamplePatchesDataFunction()
{
  delete whiteningFunction;
}

void MNISTSamplePatchesDataFunction::configurePolicy(const Matrix_t& tmpX, Matrix_t& X,
    const Matrix_t& tmpY, Matrix_t& Y)
{
  std::random_device rd;
  std::mt19937 gen(rd());
  const int imgWidth = sqrt(tmpX.cols());
  std::uniform_int_distribution<> disPatch(0, imgWidth - patchWidth); //[a, b]
  std::uniform_int_distribution<> disRows(0, tmpX.rows() - 1); //[a, b]

  X.setZero(numPatches, pow(patchWidth, 2));
#pragma omp parallel for
  for (int i = 0; i < numPatches; ++i)
  {
    int x, y, img;
#pragma omp critical
    {
      x = disPatch(gen);
      y = disPatch(gen);
      img = disRows(gen);
//      std::cout << omp_get_thread_num() << " img: " << img << " x: " << x << " y: " << y
//          << std::endl;
    }

    Vector_t row = tmpX.row(img);
    Eigen::Map<Matrix_t> Patch(row.data(), imgWidth, imgWidth);
    Matrix_t P = Patch.block(y, x, patchWidth, patchWidth);
    Eigen::Map<Vector_t> p(P.data(), pow(patchWidth, 2));
    X.row(i) = p.transpose();
  }

  X = whiteningFunction->gen(X);

  // fixme: unit ball
}


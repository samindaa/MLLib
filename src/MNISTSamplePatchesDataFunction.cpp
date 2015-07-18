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

void MNISTSamplePatchesDataFunction::configurePolicy(const Eigen::MatrixXd& tmpX,
    Eigen::MatrixXd& X, const Eigen::MatrixXd& tmpY, Eigen::MatrixXd& Y)
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

    Eigen::VectorXd row = tmpX.row(img);
    Eigen::Map<Eigen::MatrixXd> Patch(row.data(), imgWidth, imgWidth);
    Eigen::MatrixXd P = Patch.block(y, x, patchWidth, patchWidth);
    Eigen::Map<Eigen::VectorXd> p(P.data(), pow(patchWidth, 2));
    X.row(i) = p.transpose();
  }

  X = whiteningFunction->gen(X);
}


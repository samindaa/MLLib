/*
 * MNISTDataBinaryDigitsFunction.cpp
 *
 *  Created on: Jun 14, 2015
 *      Author: sam
 */

#include "MNISTDataBinaryDigitsFunction.h"

MNISTDataBinaryDigitsFunction::MNISTDataBinaryDigitsFunction(const bool& targetOneOfKCoding) :
    targetOneOfKCoding(targetOneOfKCoding)
{
}

void MNISTDataBinaryDigitsFunction::configurePolicy(const Eigen::MatrixXd& tmpX, Eigen::MatrixXd& X,
    const Eigen::MatrixXd& tmpY, Eigen::MatrixXd& Y)
{
  int numberOfPolicyRows = 0;

//  omp_set_num_threads(NUMBER_OF_OPM_THREADS);
#pragma omp parallel for reduction(+:numberOfPolicyRows)
  for (int i = 0; i < tmpY.rows(); ++i)
  {
    if (tmpY(i, 0) == 1 || tmpY(i, 1) == 1)
      ++numberOfPolicyRows;
  }

  std::cout << "numberOfPolicyRows: " << numberOfPolicyRows << std::endl;
  X.setZero(numberOfPolicyRows, tmpX.cols());
  if (targetOneOfKCoding)
    Y.setZero(numberOfPolicyRows, 2);
  else
    Y.setZero(numberOfPolicyRows, 1); //<< binary: 0 or 1

  int rowCounter = 0;

  for (int i = 0; i < tmpY.rows(); ++i)
  {
    if (tmpY(i, 0) == 1)
    {
      X.row(rowCounter) = tmpX.row(i);
      Y(rowCounter, 0) = tmpY(i, 0);
      ++rowCounter;
    }
    else if (tmpY(i, 1) == 1)
    {
      X.row(rowCounter) = tmpX.row(i);
      Y(rowCounter, targetOneOfKCoding) = tmpY(i, 1);
      ++rowCounter;
    }
  }

  assert(rowCounter == numberOfPolicyRows);
}


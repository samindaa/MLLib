/*
 * MNISTSamplePatchesUnlabeledDataFunction.cpp
 *
 *  Created on: Jul 21, 2015
 *      Author: sam
 */

#include "MNISTSamplePatchesUnlabeledDataFunction.h"

MNISTSamplePatchesUnlabeledDataFunction::MNISTSamplePatchesUnlabeledDataFunction(
    const int& numPatches, const int& patchWidth) :
    MNISTSamplePatchesDataFunction(numPatches, patchWidth)
{
}

MNISTSamplePatchesUnlabeledDataFunction::~MNISTSamplePatchesUnlabeledDataFunction()
{
}

void MNISTSamplePatchesUnlabeledDataFunction::configurePolicy(const Matrix_t& tmpX, Matrix_t& outX,
    const Matrix_t& tmpY, Matrix_t& outY)
{
  std::vector<int>* threadVectors = nullptr;
  std::vector<int> exclusivePrefixSum;
  int nbThreads;
  int nbPolicyRows = 0;
  // unlabeled data on 5 -- 9 from the first 50000
  const int top50000Rows = tmpY.rows() - 10000;
  Matrix_t newTmpX;

#pragma omp parallel
  {
    nbThreads = omp_get_num_threads();
    const int threadIdx = omp_get_thread_num();

#pragma omp single
    {
      threadVectors = new std::vector<int>[nbThreads];
    }

    Eigen::MatrixXf::Index maxIndex = std::numeric_limits<int>::max();
#pragma omp for reduction(+:nbPolicyRows)
    for (int i = 0; i < top50000Rows; ++i)
    {
      tmpY.row(i).maxCoeff(&maxIndex);
      if (maxIndex >= 5)
      {
        ++nbPolicyRows;
        threadVectors[threadIdx].push_back(i);
      }
    }

#pragma omp barrier

#pragma omp single
    {
      std::cout << "nbPolicyRows: " << nbPolicyRows << std::endl;
      newTmpX.setZero(nbPolicyRows, tmpX.cols());

      exclusivePrefixSum.push_back(0);
      for (int i = 1; i < nbThreads; ++i)
        exclusivePrefixSum.push_back(exclusivePrefixSum[i - 1] + threadVectors[i - 1].size());
    }

    const std::vector<int>& threadVector = threadVectors[threadIdx];
    for (size_t i = 0; i < threadVector.size(); ++i)
      newTmpX.row(i + exclusivePrefixSum[threadIdx]) = tmpX.row(threadVector[i]);
  }

  delete[] threadVectors;
  MNISTSamplePatchesDataFunction::configurePolicy(newTmpX, outX, tmpY, outY);
}


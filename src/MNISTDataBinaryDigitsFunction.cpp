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

void MNISTDataBinaryDigitsFunction::configurePolicy(const Matrix_t& tmpX, Matrix_t& X,
    const Matrix_t& tmpY, Matrix_t& Y)
{
  // Lets be smart and efficient
  std::vector<int>* threadVectors = nullptr;
  std::vector<int> exclusivePrefixSum;
  int nbThreads;
  int nbPolicyRows = 0;

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
    for (int i = 0; i < tmpY.rows(); ++i)
    {
      tmpY.row(i).maxCoeff(&maxIndex);
      if (maxIndex <= 1)
      {
        ++nbPolicyRows;
        threadVectors[threadIdx].push_back(i);
      }
    }

#pragma omp barrier

#pragma omp single
    {
      std::cout << "nbPolicyRows: " << nbPolicyRows << std::endl;
      X.setZero(nbPolicyRows, tmpX.cols());
      if (targetOneOfKCoding)
        Y.setZero(nbPolicyRows, 2);
      else
        Y.setZero(nbPolicyRows, 1); //<< binary: 0 or 1

      int threadAccRows = 0;
      for (int i = 0; i < nbThreads; ++i)
      {
        //std::cout << "i: " << i << " size: " << threadVectors[i].size() << std::endl;
        threadAccRows += threadVectors[i].size();
      }

      assert(nbPolicyRows == threadAccRows);

      exclusivePrefixSum.push_back(0);
      for (int i = 1; i < nbThreads; ++i)
        exclusivePrefixSum.push_back(exclusivePrefixSum[i - 1] + threadVectors[i - 1].size());

      // exclusive prefix sum
      /*std::cout << exclusivePrefixSum.size() << std::endl;
       for (int i = 0; i < nbThreads; ++i)
       std::cout << "i: " << i << " size: " << exclusivePrefixSum[i] << std::endl;
       */
    }

    const std::vector<int>& threadVector = threadVectors[threadIdx];
    for (size_t i = 0; i < threadVector.size(); ++i)
    {
      if (tmpY(threadVector[i], 0) == 1)
      {
        X.row(i + exclusivePrefixSum[threadIdx]) = tmpX.row(threadVector[i]);
        Y(i + exclusivePrefixSum[threadIdx], 0) = tmpY(threadVector[i], 0);
      }
      else if (tmpY(threadVector[i], 1) == 1)
      {
        X.row(i + exclusivePrefixSum[threadIdx]) = tmpX.row(threadVector[i]);
        Y(i + exclusivePrefixSum[threadIdx], targetOneOfKCoding) = tmpY(threadVector[i], 1);
      }
    }
  }

  delete[] threadVectors;

}


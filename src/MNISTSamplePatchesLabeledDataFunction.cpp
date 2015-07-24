/*
 * MNISTSamplePatchesLabeledDataFunction.cpp
 *
 *  Created on: Jul 21, 2015
 *      Author: sam
 */

#include "MNISTSamplePatchesLabeledDataFunction.h"

MNISTSamplePatchesLabeledDataFunction::MNISTSamplePatchesLabeledDataFunction(
    ConvolutionFunction* convolutionFunction, PoolingFunction* poolingFunction,
    const Eigen::Vector2i& convolutionFunctionConfig, const int& numFilters, const int& poolDim,
    const int& outputDim) :
    convolutionFunction(convolutionFunction), poolingFunction(poolingFunction), //
    convolutionFunctionConfig(convolutionFunctionConfig), numFilters(numFilters), poolDim(poolDim), //
    outputDim(outputDim)
{
}

void MNISTSamplePatchesLabeledDataFunction::update(const Matrix_t& tmpX, const Matrix_t& tmpY,
    Matrix_t& X, Matrix_t& Y, const int& index, const int& size)
{

  // Label data from 0 -- 4
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
    for (int i = index; i < index + size; ++i)
    {
      tmpY.row(i).maxCoeff(&maxIndex);
      if (maxIndex <= 4)
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
      Y.setZero(nbPolicyRows, 5);

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
      /* std::cout << exclusivePrefixSum.size() << std::endl;
       for (int i = 0; i < nbThreads; ++i)
       std::cout << "i: " << i << " size: " << exclusivePrefixSum[i] << " t_size: "
       << threadVectors[i].size() << std::endl;*/

    }

    const std::vector<int>& threadVector = threadVectors[threadIdx];
    for (size_t i = 0; i < threadVector.size(); ++i)
    {
      tmpY.row(threadVector[i]).maxCoeff(&maxIndex);
      X.row(i + exclusivePrefixSum[threadIdx]) = tmpX.row(threadVector[i]);
      Y(i + exclusivePrefixSum[threadIdx], maxIndex) = tmpY(threadVector[i], maxIndex);
    }
  }

  delete[] threadVectors;
}

void MNISTSamplePatchesLabeledDataFunction::configurePolicy(const Matrix_t& tmpX, Matrix_t& X,
    const Matrix_t& tmpY, Matrix_t& Y)
{

  update(tmpX, tmpY, trainingX, trainingY, 50000, 5000);
  update(tmpX, tmpY, testingX, testingY, 55000, 5000);

  Convolutions* activations = convolutionFunction->conv(trainingX, convolutionFunctionConfig);
  Poolings* activationsPooled = poolingFunction->pool(activations, poolDim);
  trainingX = Matrix_t::Zero(activationsPooled->unordered_map.size(),
      std::pow(outputDim, 2) * numFilters);

  for (auto iter = activationsPooled->unordered_map.begin();
      iter != activationsPooled->unordered_map.end(); ++iter)
    activationsPooled->toMatrix(trainingX, iter->first);

  convolutionFunction->clear();
  poolingFunction->clear();

  activations = convolutionFunction->conv(testingX, convolutionFunctionConfig);
  activationsPooled = poolingFunction->pool(activations, poolDim);
  testingX = Matrix_t::Zero(activationsPooled->unordered_map.size(),
      std::pow(outputDim, 2) * numFilters);

  for (auto iter = activationsPooled->unordered_map.begin();
      iter != activationsPooled->unordered_map.end(); ++iter)
    activationsPooled->toMatrix(testingX, iter->first);

  std::cout << "MNISTSamplePatchesLabeledDataFunction:" << std::endl;
  std::cout << trainingX.rows() << "x" << trainingX.cols() << std::endl;
  std::cout << trainingY.rows() << "x" << trainingY.cols() << std::endl;

  std::cout << testingX.rows() << "x" << testingX.cols() << std::endl;
  std::cout << testingY.rows() << "x" << testingY.cols() << std::endl;


}


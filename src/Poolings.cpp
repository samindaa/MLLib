/*
 * Poolings.cpp
 *
 *  Created on: Jul 1, 2015
 *      Author: sam
 */

#include "Poolings.h"

Poolings::Poolings(const int& numFilters, const int& outputDim) :
    numFilters(numFilters), outputDim(outputDim)
{
}

Poolings::~Poolings()
{
  for (auto iter = unordered_map_sensitivities.begin(); iter != unordered_map_sensitivities.end();
      ++iter)
  {
    for (auto iter2 = iter->second.begin(); iter2 != iter->second.end(); ++iter2)
      delete iter2->second;
  }
}

/*Eigen::VectorXd toVector(const int& imageIndex)
 {
 Eigen::VectorXd vF(std::pow(outputDim, 2) * numFilters);
 for (auto iter = pooling[imageIndex].begin(); iter != pooling[imageIndex].end(); ++iter)
 vF.segment(iter->first * std::pow(outputDim, 2), std::pow(outputDim, 2)) = //
 Eigen::Map<Eigen::VectorXd>(iter->second->X.data(), iter->second->X.size());
 return vF;
 }
 */
void Poolings::toMatrix(Eigen::MatrixXd& ActivationsPooled, const int& imageIndex)
{
  //Eigen::VectorXd vF(std::pow(outputDim, 2) * numFilters);
  const int outputDim2 = std::pow(outputDim, 2);
  for (auto iter = unordered_map[imageIndex].begin(); iter != unordered_map[imageIndex].end();
      ++iter)
    ActivationsPooled.row(imageIndex).segment(iter->first * outputDim2, outputDim2) = //
        Eigen::Map<Eigen::VectorXd>(iter->second->X.data(), iter->second->X.size());
  //return vF;
}

void Poolings::toPoolingSensitivities(const Eigen::MatrixXd& PoolingDelta)
{
  // unpooling operation
  // fixme: parallel
  const int outputDim2 = std::pow(outputDim, 2);
  for (int i = 0; i < PoolingDelta.rows(); ++i)
  {
    Eigen::VectorXd vec = PoolingDelta.row(i);
    for (int f = 0; f < numFilters; ++f)
    {

      Pooling* pooling = nullptr;
      auto poolingIter = unordered_map_sensitivities[i].find(f);
      if (poolingIter != unordered_map_sensitivities[i].end())
        pooling = poolingIter->second;
      else
      {
        pooling = new Pooling;
        unordered_map_sensitivities[i].insert(std::make_pair(f, pooling));
      }

      pooling->X = Eigen::Map<Eigen::MatrixXd>(vec.data() + f * outputDim2, //
      outputDim, outputDim);

    }

  }
}


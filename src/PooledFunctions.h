/*
 * PooledFunctions.h
 *
 *  Created on: Jun 30, 2015
 *      Author: sam
 */

#ifndef POOLEDFUNCTIONS_H_
#define POOLEDFUNCTIONS_H_

#include "PooledFunction.h"
#include "EigenFunction.h"
#include <unordered_map>

class PooledFunctions: public EigenFunction
{
  private:
    int numFilters;
    int outputDim;

  public:
    typedef std::unordered_map<int, std::unordered_map<int, PooledFunction*>> Pooling;
    Pooling pooling;

    PooledFunctions(const int& numFilters, const int& outputDim) :
        numFilters(numFilters), outputDim(outputDim)
    {
    }

    Eigen::VectorXd flatten(const int& imageIndex)
    {
      Eigen::VectorXd vF(std::pow(outputDim, 2) * numFilters);
      for (auto iter = pooling[imageIndex].begin(); iter != pooling[imageIndex].end(); ++iter)
        vF.segment(iter->first * std::pow(outputDim, 2), std::pow(outputDim, 2)) = //
            Eigen::Map<Eigen::VectorXd>(iter->second->X.data(), iter->second->X.size());
      return vF;
    }

};

#endif /* POOLEDFUNCTIONS_H_ */

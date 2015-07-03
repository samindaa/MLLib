/*
 * Poolings.h
 *
 *  Created on: Jun 30, 2015
 *      Author: sam
 */

#ifndef POOLINGS_H_
#define POOLINGS_H_

#include "EigenFunction.h"
#include <unordered_map>
#include "Pooling.h"

class Poolings: public EigenFunction
{
  private:
    int numFilters;
    int outputDim;

  public:
    std::unordered_map<int, std::unordered_map<int, Pooling*>> unordered_map;
    std::unordered_map<int, std::unordered_map<int, Pooling*>> unordered_map_sensitivities;

    Poolings(const int& numFilters, const int& outputDim);
    ~Poolings();

    void toMatrix(Eigen::MatrixXd& ActivationsPooled, const int& imageIndex);
    void toPoolingSensitivities(const Eigen::MatrixXd& PoolingDelta);
};

#endif /* POOLINGS_H_ */

/*
 * NaturalImageDataFunction.h
 *
 *  Created on: Jul 16, 2015
 *      Author: sam
 */

#ifndef NATURALIMAGEDATAFUNCTION_H_
#define NATURALIMAGEDATAFUNCTION_H_

#include <unordered_map>
#include "DataFunction.h"
#include "EigenMatrixXd.h"
#include "WhiteningFunction.h"

class NaturalImageDataFunction: public DataFunction
{
  private:
    int numPatches;
    int patchWidth;
    Config whiteningConfig;
    WhiteningFunction* whiteningFunction;
    std::unordered_map<int, EigenMatrixXd*> unordered_map;

  public:
    NaturalImageDataFunction(const int& numPatches, const int& patchWidth);
    ~NaturalImageDataFunction();

    void configure(Config* config);
};

#endif /* NATURALIMAGEDATAFUNCTION_H_ */

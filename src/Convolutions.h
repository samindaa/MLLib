/*
 * Convolutions.h
 *
 *  Created on: Jun 30, 2015
 *      Author: sam
 */

#ifndef CONVOLUTIONS_H_
#define CONVOLUTIONS_H_

#include <unordered_map>
#include "EigenFunction.h"
#include "Convolution.h"

class Convolutions: public EigenFunction
{
  public:
    std::unordered_map<int, std::unordered_map<int, Convolution*>> unordered_map;
};

#endif /* CONVOLUTIONS_H_ */

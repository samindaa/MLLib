/*
 * ConvolutedFunctions.h
 *
 *  Created on: Jun 30, 2015
 *      Author: sam
 */

#ifndef CONVOLUTEDFUNCTIONS_H_
#define CONVOLUTEDFUNCTIONS_H_

#include "ConvolutedFunction.h"
#include <unordered_map>

class ConvolutedFunctions
{
  public:
    typedef std::unordered_map<int, std::unordered_map<int, ConvolutedFunction*>> Convolutions;
    Convolutions convolutions;
};

#endif /* CONVOLUTEDFUNCTIONS_H_ */

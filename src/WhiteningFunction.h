/*
 * WhiteningFunction.h
 *
 *  Created on: Jul 13, 2015
 *      Author: sam
 */

#ifndef WHITENINGFUNCTION_H_
#define WHITENINGFUNCTION_H_

#include "EigenFunction.h"
#include "DataFunction.h"
#include "Config.h"

class WhiteningFunction: public EigenFunction
{
  private:
    Config* config;

  public:
    WhiteningFunction(Config* config);
    ~WhiteningFunction();
    Matrix_t gen(const Matrix_t& X);
};

#endif /* WHITENINGFUNCTION_H_ */

/*
 * DebugHelper.h
 *
 *  Created on: Jul 23, 2015
 *      Author: sam
 */

#ifndef DEBUGHELPER_H_
#define DEBUGHELPER_H_

#include "EigenFunction.h"
#include <iostream>

class DebugHelper: public EigenFunction
{
  public:
    static void writeMatrixTopRows(const Matrix_t& M, const int& topRows, const std::string& fname);
    static void writeMatrixBottomRows(const Matrix_t& M, const int& bottomRows,
        const std::string& fname);

};

#endif /* DEBUGHELPER_H_ */

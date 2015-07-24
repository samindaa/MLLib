/*
 * MatrixToLatex.h
 *
 *  Created on: Jul 22, 2015
 *      Author: sam
 */

#ifndef MATRIXTOLATEX_H_
#define MATRIXTOLATEX_H_

#include <iostream>
#include "EigenFunction.h"

class MatrixToLatex
{
  public:
    static void toLatex(const Matrix_t& M);
};

#endif /* MATRIXTOLATEX_H_ */

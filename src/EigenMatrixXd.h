/*
 * EigenMatrixXd.h
 *
 *  Created on: Jul 16, 2015
 *      Author: sam
 */

#ifndef EIGENMATRIXXD_H_
#define EIGENMATRIXXD_H_

#include "EigenFunction.h"

// Wrapper
class EigenMatrixXd: public EigenFunction
{
  public:
    Matrix_t X;
};

#endif /* EIGENMATRIXXD_H_ */

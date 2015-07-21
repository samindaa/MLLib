/*
 * MNISTDataBinaryDigitsFunction.h
 *
 *  Created on: Jun 14, 2015
 *      Author: sam
 */

#ifndef MNISTDATABINARYDIGITSFUNCTION_H_
#define MNISTDATABINARYDIGITSFUNCTION_H_

#include "MNISTDataFunction.h"

class MNISTDataBinaryDigitsFunction: public MNISTDataFunction
{
  private:
    bool targetOneOfKCoding;

  public:
    MNISTDataBinaryDigitsFunction(const bool& targetOneOfKCoding = false);

    void configurePolicy(const Matrix_t& tmpX, Matrix_t& X, const Matrix_t& tmpY, Matrix_t& Y);
};

#endif /* MNISTDATABINARYDIGITSFUNCTION_H_ */

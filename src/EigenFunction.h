/*
 * EigenFunction.h
 *
 *  Created on: Jun 27, 2015
 *      Author: sam
 */

#ifndef EIGENFUNCTION_H_
#define EIGENFUNCTION_H_

#include <omp.h>
#include "Eigen/Dense"
#include "unsupported/Eigen/KroneckerProduct"

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Matrix_t;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> Vector_t;
typedef Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> Array_t;

class EigenFunction
{
};

#endif /* EIGENFUNCTION_H_ */

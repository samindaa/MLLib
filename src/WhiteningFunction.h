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
    Eigen::MatrixXd gen(const Eigen::MatrixXd& X);

  private:
    void zeroMean(Eigen::MatrixXd& X);
    void pcaWhitening(Eigen::MatrixXd& X);
    void zcaWhitening(Eigen::MatrixXd& X);

};

#endif /* WHITENINGFUNCTION_H_ */

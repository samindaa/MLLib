/*
 * DataFunction.h
 *
 *  Created on: Jun 12, 2015
 *      Author: sam
 */

#ifndef DATAFUNCTION_H_
#define DATAFUNCTION_H_

#include "EigenFunction.h"
#include "ConfigurationDescription.h"

class DataFunction: public EigenFunction
{
  protected:
    Eigen::MatrixXd trainingX;
    Eigen::MatrixXd trainingY;

    Eigen::MatrixXd testingX;
    Eigen::MatrixXd testingY;

    // mean and stdd calculated from the training data
    Eigen::VectorXd mean;
    Eigen::VectorXd stdd;

  public:
    DataFunction();
    virtual ~DataFunction();

    virtual void configure(const ConfigurationDescription* configuration) =0;

    Eigen::MatrixXd& getTrainingX();
    Eigen::MatrixXd& getTestingX();
    Eigen::MatrixXd& getTrainingY();
    Eigen::MatrixXd& getTestingY();

  protected:
    void trainingMeanAndStdd();
    void datasetsMeanNormalize(const double& stdd_offset = 0.0f);
    void datasetsSetBias();

};

#endif /* DATAFUNCTION_H_ */

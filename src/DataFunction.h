/*
 * DataFunction.h
 *
 *  Created on: Jun 12, 2015
 *      Author: sam
 */

#ifndef DATAFUNCTION_H_
#define DATAFUNCTION_H_

#include "EigenFunction.h"
#include "Config.h"

class DataFunction: public EigenFunction
{
  protected:
    Matrix_t trainingX;
    Matrix_t trainingY;

    Matrix_t testingX;
    Matrix_t testingY;

    // mean and stdd calculated from the training data
    Vector_t mean;
    Vector_t stdd;

  public:
    DataFunction();
    virtual ~DataFunction();

    virtual void configure(Config* config) =0;

    Matrix_t& getTrainingX();
    Matrix_t& getTestingX();
    Matrix_t& getTrainingY();
    Matrix_t& getTestingY();

  protected:
    void trainingMeanAndStdd();
    void datasetsMeanNormalize(const double& stdd_offset = 0.0f);
    void datasetsSetBias();

};

#endif /* DATAFUNCTION_H_ */

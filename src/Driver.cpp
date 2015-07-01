/*
 * Driver.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: sam
 */

#ifndef DRIVER_CPP_
#define DRIVER_CPP_

#include "Driver.h"
#include <iostream>
#include <fstream>

Driver::Driver(ConfigurationDescription* configuration, DataFunction* dataFunction,
    CostFunction* costFunction, Optimizer* optimizer) :
    configuration(configuration), dataFunction(dataFunction), costFunction(costFunction), //
    optimizer(optimizer)
{
}

Driver::~Driver()
{
}

void Driver::drive()
{
  dataFunction->configure(configuration);
  Eigen::VectorXd theta = costFunction->configure(dataFunction->getTrainingX(),
      dataFunction->getTrainingY());

  costFunction->getNumGrad(theta, dataFunction->getTrainingX(), dataFunction->getTrainingY(), 5);

//  return;

  optimizer->optimize(theta, dataFunction, costFunction); //<<fixme

  std::cout << "training_accuracy: "
      << costFunction->accuracy(theta, dataFunction->getTrainingX(), dataFunction->getTrainingY())
      << std::endl;
  std::cout << "testing_accuracy: "
      << costFunction->accuracy(theta, dataFunction->getTestingX(), dataFunction->getTestingY())
      << std::endl;
}

#endif /* DRIVER_CPP_ */

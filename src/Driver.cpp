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

Driver::Driver(Config* config, DataFunction* dataFunction, CostFunction* costFunction,
    Optimizer* optimizer) :
    config(config), dataFunction(dataFunction), costFunction(costFunction), optimizer(optimizer)
{
}

Driver::~Driver()
{
}

void Driver::drive()
{
  dataFunction->configure(config);
  Eigen::VectorXd theta = costFunction->configure(dataFunction->getTrainingX(),
      dataFunction->getTrainingY());

  if (config->getValue("numGrd", false))
    costFunction->getNumGrad(theta, dataFunction->getTrainingX(), dataFunction->getTrainingY(), 5);

  optimizer->optimize(theta, dataFunction, costFunction); //<<fixme

  if (config->getValue("training_accuracy", false))
    std::cout << "training_accuracy: "
        << costFunction->accuracy(theta, dataFunction->getTrainingX(), dataFunction->getTrainingY())
        << std::endl;

  std::cout << "testing_accuracy: "
      << costFunction->accuracy(theta, dataFunction->getTestingX(), dataFunction->getTestingY())
      << std::endl;
}

#endif /* DRIVER_CPP_ */

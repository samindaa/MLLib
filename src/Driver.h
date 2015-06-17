/*
 * Driver.h
 *
 *  Created on: Jun 12, 2015
 *      Author: sam
 */

#ifndef DRIVER_H_
#define DRIVER_H_

#include "DataFunction.h"
#include "CostFunction.h"
#include "Optimizer.h"

class Driver
{
  private:
    ConfigurationDescription* configuration;
    DataFunction* dataFunction;
    CostFunction* costFunction;
    Optimizer* optimizer;

  public:
    Driver(ConfigurationDescription* configuration, DataFunction* dataFunction,
        CostFunction* costFunction, Optimizer* optimizer);
    ~Driver();
    void drive();

};

#endif /* DRIVER_H_ */

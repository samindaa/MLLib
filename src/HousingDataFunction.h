/*
 * HousingDataFunction.h
 *
 *  Created on: Jun 12, 2015
 *      Author: sam
 */

#ifndef HOUSINGDATAFUNCTION_H_
#define HOUSINGDATAFUNCTION_H_

#include "DataFunction.h"

class HousingDataFunction: public DataFunction
{
  public:
    void configure(const ConfigurationDescription* configuration);
};

#endif /* HOUSINGDATAFUNCTION_H_ */

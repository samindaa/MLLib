/*
 * IEEEHumanDataFunction.h
 *
 *  Created on: Jun 16, 2015
 *      Author: sam
 */

#ifndef IEEEHUMANDATAFUNCTION_H_
#define IEEEHUMANDATAFUNCTION_H_

#include "DataFunction.h"
#include <vector>
#include <iostream>

class IEEEHumanDataFunction: public DataFunction
{
  private:
    struct Meta
    {
        std::vector<double> fdata;
        int target;
    };

  public:
    void configure(Config* config);

  private:
    void read(const std::vector<std::string>& filevector, std::vector<Meta>& datavector,
        const int& targetIndex);
};

#endif /* IEEEHUMANDATAFUNCTION_H_ */

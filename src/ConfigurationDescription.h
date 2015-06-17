/*
 * ConfigurationDescription.h
 *
 *  Created on: Jun 13, 2015
 *      Author: sam
 */

#ifndef CONFIGURATIONDESCRIPTION_H_
#define CONFIGURATIONDESCRIPTION_H_

#include <iostream>
#include <unordered_map>

class ConfigurationDescription
{
  public:
    //<< (key, value)
    typedef std::unordered_map<std::string, std::string> Config;
    Config config;

    bool meanStddNormalize;
    bool addBiasTerm;

    ConfigurationDescription() :
        meanStddNormalize(true), addBiasTerm(true)
    {
    }
};

#endif /* CONFIGURATIONDESCRIPTION_H_ */

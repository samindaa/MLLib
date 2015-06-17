/*
 * IEEEHumanDataFunction.cpp
 *
 *  Created on: Jun 16, 2015
 *      Author: sam
 */

#include "IEEEHumanDataFunction.h"
#include <fstream>
#include <sstream>
#include <cstring>
#include <algorithm>

void IEEEHumanDataFunction::configure(const ConfigurationDescription* configuration)
{
  srand(0);

  std::string base = "/home/sam/Projects/muenergia/datasets/ieee_sensor_journal/ieee_human/";

  std::string true1 = base + "event_data.data-fall-forward.txt";
  std::string true2 = base + "event_data.data-fall-backward.txt";
  std::string true3 = base + "event_data.data-fall-left.txt";
  std::string true4 = base + "event_data.data-fall-right.txt";

  std::string negg1 = base + "event_data.data-walk-forward.txt";
  std::string negg2 = base + "event_data.data-walk-left.txt";
  std::string negg3 = base + "event_data.data-walk-right.txt";
  std::string negg4 = base + "event_data.data-walk-backward.txt";

  std::vector<std::vector<std::string>> targets;
  std::vector<std::vector<Meta>> metas;

  // At most 4
  std::vector<Meta> data1;
  std::vector<Meta> data2;
  std::vector<Meta> data3;
  std::vector<Meta> data4;

  std::vector<int> trainingSamplesVector;

  ConfigurationDescription::Config::const_iterator iter = configuration->config.find("exp");

  if (iter == configuration->config.end())
  {
    std::cerr << "Configuration is not set" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (iter->second == "__L1__")
  {
    std::vector<std::string> target1 { true1, true2, true3, true4 };
    std::vector<std::string> target2 { negg1, negg2, negg3, negg4 };

    targets.push_back(target1);
    targets.push_back(target2);

    metas.push_back(data1);
    metas.push_back(data2);

    trainingSamplesVector.push_back(18);
    trainingSamplesVector.push_back(30);
  }
  else if (iter->second == "__L2_true__")
  {
    std::vector<std::string> target1 { true1 };
    std::vector<std::string> target2 { true2 };
    std::vector<std::string> target3 { true3 };
    std::vector<std::string> target4 { true4 };

    targets.push_back(target1);
    targets.push_back(target2);
    targets.push_back(target3);
    targets.push_back(target4);

    metas.push_back(data1);
    metas.push_back(data2);
    metas.push_back(data3);
    metas.push_back(data4);

    trainingSamplesVector.push_back(3);
    trainingSamplesVector.push_back(3);
    trainingSamplesVector.push_back(3);
    trainingSamplesVector.push_back(3);
  }
  else if (iter->second == "__L2_nn_negg__")
  {
    std::vector<std::string> target1 { negg1 };
    std::vector<std::string> target2 { negg2 };
    std::vector<std::string> target3 { negg3 };
    std::vector<std::string> target4 { negg4 };

    targets.push_back(target1);
    targets.push_back(target2);
    targets.push_back(target3);
    targets.push_back(target4);

    metas.push_back(data1);
    metas.push_back(data2);
    metas.push_back(data3);
    metas.push_back(data4);

    trainingSamplesVector.push_back(12);
    trainingSamplesVector.push_back(12);
    trainingSamplesVector.push_back(12);
    trainingSamplesVector.push_back(12);
  }
  else
  {
    std::cerr << "Option: " << iter->second << " is not found!" << std::endl;
    exit(EXIT_FAILURE);
  }

  assert(targets.size() == metas.size());
  assert(trainingSamplesVector.size() == targets.size());

  for (size_t i = 0; i < targets.size(); i++)
  {
    read(targets[i], metas[i], i);
    std::cout << "\t metas[" << i << "]: " << metas[i].size() << std::endl;
  }

  for (int i = 0; i < 20; ++i)
  {
    for (size_t j = 0; j < targets.size(); ++j)
      std::random_shuffle(metas[j].begin(), metas[j].end());
  }

  const int numberOfTrainingSamples = std::accumulate(trainingSamplesVector.begin(),
      trainingSamplesVector.end(), 0);

  int numberOfSamples = 0;
  for (size_t i = 0; i < targets.size(); i++)
    numberOfSamples += metas[i].size();

  const int numberOfTestingSamples = numberOfSamples - numberOfTrainingSamples;

  std::cout << "numberOfTrainingSamples:  " << numberOfTrainingSamples << std::endl;
  std::cout << "numberOfTestingSamples: " << numberOfTestingSamples << std::endl;
  std::cout << "numberOfSamples:  " << numberOfSamples << std::endl;

  trainingX.setZero(numberOfTrainingSamples, 6);
  trainingY.setZero(numberOfTrainingSamples, targets.size());

  testingX.setZero(numberOfTestingSamples, 6);
  testingY.setZero(numberOfTestingSamples, targets.size());

  int k = 0;
  for (size_t i = 0; i < targets.size(); ++i) // each data
  {
    std::vector<Meta>& meta = metas[i];
    for (int j = 0; j < trainingSamplesVector[i]; ++j, ++k) // samples in meta
    {
      Meta& anMeta = meta[j];
      for (int p = 0; p < (int) anMeta.fdata.size(); ++p)
        trainingX(k, p) = anMeta.fdata[p];
      trainingY(k, anMeta.target) = 1.0f;
    }
  }

  assert(k == numberOfTrainingSamples);

  k = 0;
  for (size_t i = 0; i < targets.size(); ++i) // each data
  {
    std::vector<Meta>& meta = metas[i];
    for (int j = trainingSamplesVector[i]; j < (int) meta.size(); ++j, ++k) // samples in meta
    {
      Meta& anMeta = meta[j];
      for (int p = 0; p < (int) anMeta.fdata.size(); ++p)
        testingX(k, p) = anMeta.fdata[p];
      testingY(k, anMeta.target) = 1.0f;
    }
  }

  assert(k == numberOfTestingSamples);

//  std::cout << trainingX << std::endl;
//  std::cout << trainingY << std::endl;

//  std::cout << std::endl << std::endl;

//  std::cout << testingX << std::endl;
//  std::cout << testingY << std::endl;

  trainingMeanAndStdd();
  datasetsMeanNormalize();
  datasetsSetBias();
}

void IEEEHumanDataFunction::read(const std::vector<std::string>& filevector,
    std::vector<Meta>& datavector, const int& targetIndex)
{
  for (auto& x : filevector)
  {
    std::ifstream ifs(x.c_str());
    std::string str;
    while (std::getline(ifs, str))
    {
      if (str.size() > 0)
      {
        datavector.push_back(Meta());
        std::istringstream ss(str);
        std::string token;
        while (std::getline(ss, token, ','))
        {
          std::stringstream sst(token);
          double tmp;
          sst >> tmp;
          datavector[datavector.size() - 1].fdata.push_back(tmp);
        }
        datavector[datavector.size() - 1].target = targetIndex;
      }
    }
  }
}


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

void IEEEHumanDataFunction::configure(Config* config)
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

  std::string negg5 = base + "event_data.data-marching.txt";
  std::string negg6 = base + "event_data.data-rotate-ccw.txt";
  std::string negg7 = base + "event_data.data-rotate-cw.txt";

  std::vector<std::vector<std::string>> targets;
  std::vector<std::vector<Meta>> metas;

  // At most 4
  //std::vector<Meta> data1;
  //std::vector<Meta> data2;
  //std::vector<Meta> data3;
  //std::vector<Meta> data4;

  std::vector<int> trainingSamplesVector;

  const std::string mode = config->getValue("exp", std::string("[]"));

  if (mode == "__L1__")
  {
    std::vector<std::string> target1 { true1, true2, true3, true4 };
    std::vector<std::string> target2 { negg1, negg2, negg3, negg4, negg5, negg6, negg7 };

    targets.push_back(target1);
    targets.push_back(target2);

    for (size_t i = 0; i < targets.size(); ++i)
      metas.push_back(std::vector<Meta>());

    trainingSamplesVector.push_back(20);
    trainingSamplesVector.push_back(30);
  }
  else if (mode == "__L2_true__")
  {
    std::vector<std::string> target1 { true1 };
    std::vector<std::string> target2 { true2 };
    std::vector<std::string> target3 { true3 };
    std::vector<std::string> target4 { true4 };

    targets.push_back(target1);
    targets.push_back(target2);
    targets.push_back(target3);
    targets.push_back(target4);

    for (size_t i = 0; i < targets.size(); ++i)
      metas.push_back(std::vector<Meta>());

    for (size_t i = 0; i < targets.size(); ++i)
      trainingSamplesVector.push_back(3);
  }
  else if (mode == "__L2_negg__")
  {
    std::vector<std::string> target1 { negg1 };
    std::vector<std::string> target2 { negg2 };
    std::vector<std::string> target3 { negg3 };
    std::vector<std::string> target4 { negg4 };

    std::vector<std::string> target5 { negg5 };
    std::vector<std::string> target6 { negg6 };
    std::vector<std::string> target7 { negg7 };

    targets.push_back(target1);
    targets.push_back(target2);
    targets.push_back(target3);
    targets.push_back(target4);

    targets.push_back(target5);
    targets.push_back(target6);
    targets.push_back(target7);

    for (size_t i = 0; i < targets.size(); ++i)
      metas.push_back(std::vector<Meta>());

    for (size_t i = 0; i < targets.size(); ++i)
      trainingSamplesVector.push_back(12);

  }
  else if (mode == "__LX__")
  {
    std::vector<std::string> target1 { true1 };
    std::vector<std::string> target2 { true2 };
    std::vector<std::string> target3 { true3 };
    std::vector<std::string> target4 { true4 };

    std::vector<std::string> target5 { negg1 };
    std::vector<std::string> target6 { negg2 };
    std::vector<std::string> target7 { negg3 };
    std::vector<std::string> target8 { negg4 };

    std::vector<std::string> target9 { negg5 };
    std::vector<std::string> target10 { negg6 };
    std::vector<std::string> target11 { negg7 };

    targets.push_back(target1);
    targets.push_back(target2);
    targets.push_back(target3);
    targets.push_back(target4);

    targets.push_back(target5);
    targets.push_back(target6);
    targets.push_back(target7);
    targets.push_back(target8);

    targets.push_back(target9);
    targets.push_back(target10);
    targets.push_back(target11);

    for (size_t i = 0; i < targets.size(); ++i)
      metas.push_back(std::vector<Meta>());

    for (size_t i = 0; i < 4; ++i)
      trainingSamplesVector.push_back(3);

    for (size_t i = 4; i < targets.size(); ++i)
      trainingSamplesVector.push_back(12);

  }
  else
  {
    std::cerr << "Option: " << mode << " is not found!" << std::endl;
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


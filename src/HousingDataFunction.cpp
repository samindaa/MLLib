/*
 * HousingDataFunction.cpp
 *
 *  Created on: Jun 12, 2015
 *      Author: sam
 */

#include "HousingDataFunction.h"
// [Load]
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
// [Read]
#include <sstream>
#include <algorithm>
#include <iterator>
#include <vector>
// [OMP]
#include <omp.h>

void HousingDataFunction::configure(const ConfigurationDescription* configuration)
{
  Eigen::MatrixXd TmpData;
  //<< housing dataset
  auto iter = configuration->config.find("housing.data");
  if (iter == configuration->config.end())
  {
    std::cerr << "housing.data: missing" << std::endl;
    assert(false);
  }

  std::ifstream in(iter->second.c_str());
  if (in.is_open())
  {
    std::string str;
    int nbRows = 0;
    while (std::getline(in, str))
    {
      if (str.size() == 0)
        continue;
      std::istringstream iss(str);
      std::vector<double> tokens //
      { std::istream_iterator<double> { iss }, std::istream_iterator<double> { } };
      TmpData.conservativeResize(nbRows + 1, tokens.size());
      //<< fixme
      for (size_t i = 0; i < tokens.size(); ++i)
        TmpData(nbRows, i) = tokens[i];
      ++nbRows;
    }
  }
  else
  {
    std::cerr << "file: " << iter->second << " failed" << std::endl;
    assert(false);
  }

  //std::cout << data << std::endl;

  //return;

  std::cout << "rows: " << TmpData.rows() << " cols: " << TmpData.cols() << std::endl;

  // split
  std::vector<int> rows(TmpData.rows());
  std::generate(rows.begin(), rows.end(), []
  { static int i(0);
    return i++;});

  std::random_shuffle(rows.begin(), rows.end());

  trainingX.setZero(400, TmpData.cols() - 1);
  trainingY.setZero(400, 1);

  testingX.setZero(TmpData.rows() - 400, TmpData.cols() - 1);
  testingY.setZero(TmpData.rows() - 400, 1);

//  omp_set_num_threads(NUMBER_OF_OPM_THREADS);
#pragma omp parallel for
  for (int i = 0; i < 400; ++i)
  {
    for (int j = 0; j < TmpData.cols() - 1; ++j)
      trainingX(i, j) = TmpData(rows[i], j);
    trainingY(i, 0) = TmpData(rows[i], TmpData.cols() - 1);
  }

//  omp_set_num_threads(NUMBER_OF_OPM_THREADS);
#pragma omp parallel for
  for (int i = 400; i < TmpData.rows(); ++i)
  {
    for (int j = 0; j < TmpData.cols() - 1; ++j)
      testingX(i - 400, j) = TmpData(rows[i], j);
    testingY(i - 400, 0) = TmpData(rows[i], TmpData.cols() - 1);
  }

  trainingMeanAndStdd();
  datasetsMeanNormalize();
//  datasetsSetBias();

}


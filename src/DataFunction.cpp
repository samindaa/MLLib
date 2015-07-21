/*
 * DataFunction.cpp
 *
 *  Created on: Jun 13, 2015
 *      Author: sam
 */

#include "DataFunction.h"

DataFunction::DataFunction()
{
}

DataFunction::~DataFunction()
{
}

Matrix_t& DataFunction::getTrainingX()
{
  return trainingX;
}

Matrix_t& DataFunction::getTestingX()
{
  return testingX;
}

Matrix_t& DataFunction::getTrainingY()
{
  return trainingY;
}

Matrix_t& DataFunction::getTestingY()
{
  return testingY;
}

void DataFunction::trainingMeanAndStdd()
{
  mean = trainingX.colwise().mean();
  stdd = (trainingX.rowwise() - mean.transpose()).array().square().colwise().mean().array().sqrt();
}

void DataFunction::datasetsMeanNormalize(const double& stdd_offset)
{
  std::cout << "meanNormalize: " << std::endl;
  // Mean normalize (x - mean) ./ stdd
  trainingX.rowwise() -= mean.transpose();
  trainingX *= (stdd.array() + stdd_offset).inverse().matrix().asDiagonal();

  testingX.rowwise() -= mean.transpose();
  testingX *= (stdd.array() + stdd_offset).inverse().matrix().asDiagonal();

}

void DataFunction::datasetsSetBias()
{
  std::cout << "setBias: " << std::endl;
  // add the bias term
  trainingX.conservativeResize(Eigen::NoChange, trainingX.cols() + 1);
  trainingX.col(trainingX.cols() - 1) = Vector_t::Ones(trainingX.rows());

  testingX.conservativeResize(Eigen::NoChange, testingX.cols() + 1);
  testingX.col(testingX.cols() - 1) = Vector_t::Ones(testingX.rows());

}


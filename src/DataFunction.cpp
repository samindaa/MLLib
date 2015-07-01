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

Eigen::MatrixXd& DataFunction::getTrainingX()
{
  return trainingX;
}

Eigen::MatrixXd& DataFunction::getTestingX()
{
  return testingX;
}

Eigen::MatrixXd& DataFunction::getTrainingY()
{
  return trainingY;
}

Eigen::MatrixXd& DataFunction::getTestingY()
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
  // Mean normalize (x - mean) ./ stdd
  trainingX.rowwise() -= mean.transpose();
  trainingX *= (stdd.array() + stdd_offset).inverse().matrix().asDiagonal();

  testingX.rowwise() -= mean.transpose();
  testingX *= (stdd.array() + stdd_offset).inverse().matrix().asDiagonal();

}

void DataFunction::datasetsSetBias()
{
  // add the bias term
  trainingX.conservativeResize(Eigen::NoChange, trainingX.cols() + 1);
  trainingX.col(trainingX.cols() - 1) = Eigen::VectorXd::Ones(trainingX.rows());

  testingX.conservativeResize(Eigen::NoChange, testingX.cols() + 1);
  testingX.col(testingX.cols() - 1) = Eigen::VectorXd::Ones(testingX.rows());

}


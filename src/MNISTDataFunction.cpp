/*
 * MNISTDataFunction.cpp
 *
 *  Created on: Jun 13, 2015
 *      Author: sam
 */

#include "MNISTDataFunction.h"
#include <fstream>
#include <sstream>
#include <vector>

void MNISTDataFunction::configure(Config* config)
{
  Matrix_t tmpTrX, tmpTrY, tmpTeX, tmpTeY;

  // Training
  imagesLabelsLoad(config->getValue(trainImagesKeyMnist, std::string(trainImagesKeyMnist)), //
  config->getValue(trainLabelsKeyMnist, std::string(trainLabelsKeyMnist)), //
  tmpTrX, tmpTrY, config->getValue("debugMode", false));

  //Testing
  imagesLabelsLoad(config->getValue(testImagesKeyMnist, std::string(testImagesKeyMnist)), //
  config->getValue(testLabelsKeyMnist, std::string(testLabelsKeyMnist)), //
  tmpTeX, tmpTeY, config->getValue("debugMode", false));

  if (config->getValue("configurePolicyTraining", true))
    configurePolicy(tmpTrX, trainingX, tmpTrY, trainingY);
  if (config->getValue("configurePolicyTesting", true))
    configurePolicy(tmpTeX, testingX, tmpTeY, testingY);

  if (config->getValue("trainingMeanAndStdd", true))
    trainingMeanAndStdd();
  if (config->getValue("meanStddNormalize", true))
    datasetsMeanNormalize(0.1f);
  if (config->getValue("addBiasTerm", true))
    datasetsSetBias();

  // test
  /*{
   std::stringstream ss;
   ss << "training.txt";
   std::ofstream ofs(ss.str().c_str());
   ofs << tmpTrX.topRows<1>() << std::endl;
   }*/

  /*{
   std::stringstream ss;
   ss << "testing.txt";
   std::ofstream ofs(ss.str().c_str());
   ofs << testingX.row(2031) << std::endl;
   }*/

}

void MNISTDataFunction::configurePolicy(const Matrix_t& tmpX, Matrix_t& X,
    const Matrix_t& tmpY, Matrix_t& Y)
{
  X = tmpX;
  Y = tmpY;
}

// helper function
static int reverseInt(int i)
{
  unsigned char c1, c2, c3, c4;

  c1 = i & 255;
  c2 = (i >> 8) & 255;
  c3 = (i >> 16) & 255;
  c4 = (i >> 24) & 255;

  return ((int) c1 << 24) + ((int) c2 << 16) + ((int) c3 << 8) + c4;
}

void MNISTDataFunction::imagesLabelsLoad(const std::string& imagesfilename,
    const std::string& labelsfilename, Matrix_t& images, Matrix_t& labels,
    const bool& debugMode)
{
  std::ifstream ifsImages(imagesfilename.c_str(), std::ios::binary);
  std::ifstream ifsLabels(labelsfilename.c_str(), std::ios::binary);
  if (!ifsImages.is_open() && !ifsLabels.is_open())
  {
    std::cout << "data cannot be loaded: " << imagesfilename << " " << labelsfilename << std::endl;
    exit(EXIT_FAILURE);
  }

  int magicNumberImages;
  int magicNumberLabels;
  int numberOfImages;
  int numberOfLabels;
  int numberOfRows;
  int numberOfCols;

  ifsImages.read((char*) &magicNumberImages, sizeof(magicNumberImages));
  magicNumberImages = reverseInt(magicNumberImages);

  ifsImages.read((char*) &numberOfImages, sizeof(numberOfImages));
  numberOfImages = reverseInt(numberOfImages);

  ifsImages.read((char*) &numberOfRows, sizeof(numberOfRows));
  numberOfRows = reverseInt(numberOfRows);

  ifsImages.read((char*) &numberOfCols, sizeof(numberOfCols));
  numberOfCols = reverseInt(numberOfCols);

  ifsLabels.read((char*) &magicNumberLabels, sizeof(magicNumberLabels));
  magicNumberLabels = reverseInt(magicNumberLabels);

  ifsLabels.read((char*) &numberOfLabels, sizeof(numberOfLabels));
  numberOfLabels = reverseInt(numberOfLabels);

  if (debugMode)
  {
    numberOfImages = numberOfLabels = 23;
    std::cout << "DEBUG MODE" << std::endl;
  }

  std::cout << "magicNumberImages: " << magicNumberImages << " numberOfImages: " << numberOfImages
      << " numberOfRows: " << numberOfRows << " numberOfCols: " << numberOfCols
      << " magicNumberLabels: " << magicNumberLabels << " numberOfLabels: " << numberOfLabels
      << "\n imagesfilename: " << imagesfilename << "\n  labelsfilename: " << labelsfilename
      << std::endl;

  assert(numberOfImages == numberOfLabels);

  //<< fixme: efficient?
  unsigned char temp = 0;
  images.setZero(numberOfImages, numberOfRows * numberOfCols);
  labels.setZero(numberOfLabels, 10);
  for (int i = 0; i < numberOfImages; ++i)
  {

    ifsLabels.read((char*) &temp, sizeof(temp));
    labels(i, (int) temp) = 1.0f;

    for (int r = 0; r < numberOfRows; ++r)
    {
      for (int c = 0; c < numberOfCols; ++c)
      {
        ifsImages.read((char*) &temp, sizeof(temp));
        images(i, r + c * numberOfCols) = (double) temp / 255.0f;
      }
    }
  }

}

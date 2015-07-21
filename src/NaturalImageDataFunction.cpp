/*
 * NaturalImageDataFunction.cpp
 *
 *  Created on: Jul 16, 2015
 *      Author: sam
 */

#include "NaturalImageDataFunction.h"
// [TIFF]
#include <fstream>
#include <iostream>
#include <string>
#include <cassert>
#include <random>
#include <dirent.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "tiffio.h"

NaturalImageDataFunction::NaturalImageDataFunction(const int& numPatches, const int& patchWidth) :
    numPatches(numPatches), patchWidth(patchWidth)
{
  whiteningConfig.setValue("zeroMean", true);
  whiteningConfig.setValue("pcaWhitening", true);
  whiteningConfig.setValue("zcaWhitening", true);
  whiteningConfig.setValue("epsilon", 1e-4);

  whiteningFunction = new WhiteningFunction(&whiteningConfig);
}

NaturalImageDataFunction::~NaturalImageDataFunction()
{
  for (auto iter = unordered_map.begin(); iter != unordered_map.end(); ++iter)
    delete iter->second;
  unordered_map.clear();

  delete whiteningFunction;
}

void NaturalImageDataFunction::configure(Config* config)
{
  std::string sdir = config->getValue("naturalImageDir",
      std::string("/home/sam/School/online/stanford_dl_ex/common/tiff/data"));
  DIR* dir = opendir(sdir.c_str());
  if (!dir)
  {
    std::cerr << "Error(" << errno << ") opening " << sdir << std::endl;
    exit(EXIT_FAILURE);
  }

  dirent* dirp = nullptr;
  struct stat filestat;
  int filecount = 0;

  while ((dirp = readdir(dir)))
  {
    std::string filepath = sdir + "/" + dirp->d_name;
    // If the file is a directory (or is in some way invalid) we'll skip it
    if (stat(filepath.c_str(), &filestat))
      continue;
    if (S_ISDIR(filestat.st_mode))
      continue;

    TIFF* tif = TIFFOpen(filepath.c_str(), "r");
    if (tif)
    {
      uint32 w, h;
      size_t npixels;
      uint32* raster;

      TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
      TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
      npixels = w * h;

      raster = (uint32*) _TIFFmalloc(npixels * sizeof(uint32));
      if (raster != NULL)
      {
        if (TIFFReadRGBAImage(tif, w, h, raster, 0))
        {
          EigenMatrixXd* tiffImage = new EigenMatrixXd;
          tiffImage->X.setZero(h, w);

          for (size_t c_h = 0; c_h < h; c_h++)
          {
            for (size_t c_w = 0; c_w < w; c_w++)
              tiffImage->X(c_h, c_w) = raster[(w * h) - ((c_h * w) + (w - c_w))] % 256;
          }
          unordered_map.insert(std::make_pair(filecount, tiffImage));
        }

        std::cout << "f: " << filepath << std::endl;
        ++filecount;
        _TIFFfree(raster);
      }
    }
  }

  std::cout << "filecount: " << filecount << std::endl;
  assert(filecount == (int )unordered_map.size());
  closedir(dir);

  // create patches
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> uniform_int_distribution_rows(0, filecount - 1); //[a, b]

  std::unordered_map<int, std::uniform_int_distribution<>> uniform_int_distributions;

  std::cout << "numPatches: " << numPatches << " patchWidth: " << patchWidth << std::endl;

  trainingX.setZero(numPatches, pow(patchWidth, 2));

#pragma omp parallel for
  for (int i = 0; i < numPatches; ++i)
  {
    int x, y, img;
#pragma omp critical
    {
      img = uniform_int_distribution_rows(gen);

      const int imgw = unordered_map[img]->X.cols();
      const int imgh = unordered_map[img]->X.rows();

      auto iterw = uniform_int_distributions.find(imgw);
      auto iterh = uniform_int_distributions.find(imgh);

      if (iterw == uniform_int_distributions.end())
        uniform_int_distributions.insert(
            std::make_pair(imgw, std::uniform_int_distribution<>(0, imgw - patchWidth)));
      if (iterh == uniform_int_distributions.end())
        uniform_int_distributions.insert(
            std::make_pair(imgh, std::uniform_int_distribution<>(0, imgh - patchWidth)));

      y = uniform_int_distributions[imgh](gen);
      x = uniform_int_distributions[imgw](gen);

      //std::cout << omp_get_thread_num() << " img: " << img << " x: " << x << " y: " << y
      //    << std::endl;
    }

    Matrix_t P = unordered_map[img]->X.block(y, x, patchWidth, patchWidth);
    Eigen::Map<Vector_t> p(P.data(), pow(patchWidth, 2));
    trainingX.row(i) = p.transpose();
  }

//  std::cout << "before: " << std::endl;
//  std::ofstream ofsb("before.txt");
//  ofsb << trainingX << std::endl;

  //trainingX = whiteningFunction->gen(trainingX);

//  std::cout << "after: " << std::endl;
//  std::ofstream ofsa("after.txt");
//  ofsa << trainingX << std::endl;

}


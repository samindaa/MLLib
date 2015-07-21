/*
 * MNISTDataFunction.h
 *
 *  Created on: Jun 13, 2015
 *      Author: sam
 */

#ifndef MNISTDATAFUNCTION_H_
#define MNISTDATAFUNCTION_H_

#include "DataFunction.h"

#define trainImagesKeyMnist  "train-images-idx3-ubyte"
#define trainLabelsKeyMnist  "train-labels-idx1-ubyte"
#define testImagesKeyMnist   "t10k-images-idx3-ubyte"
#define testLabelsKeyMnist   "t10k-labels-idx1-ubyte"

class MNISTDataFunction: public DataFunction
{
  public:
    void configure(Config* config);

  protected:
    virtual void configurePolicy(const Matrix_t& tmpX, Matrix_t& X,
        const Matrix_t& tmpY, Matrix_t& Y);

  private:
    void imagesLabelsLoad(const std::string& imagesfilename, const std::string& labelsfilename,
        Matrix_t& images, Matrix_t& labels, const bool& debugMode);
};

#endif /* MNISTDATAFUNCTION_H_ */

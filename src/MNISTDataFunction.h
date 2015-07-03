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
    virtual void configurePolicy(const Eigen::MatrixXd& tmpX, Eigen::MatrixXd& X,
        const Eigen::MatrixXd& tmpY, Eigen::MatrixXd& Y);

  private:
    void imagesLabelsLoad(const std::string& imagesfilename, const std::string& labelsfilename,
        Eigen::MatrixXd& images, Eigen::MatrixXd& labels, const bool& debugMode);
};

#endif /* MNISTDATAFUNCTION_H_ */

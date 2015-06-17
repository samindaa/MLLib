/*
 * MNISTDataFunction.h
 *
 *  Created on: Jun 13, 2015
 *      Author: sam
 */

#ifndef MNISTDATAFUNCTION_H_
#define MNISTDATAFUNCTION_H_

#include "DataFunction.h"

class MNISTDataFunction: public DataFunction
{
  public:
    const std::string trainImagesKey { "train-images-idx3-ubyte" };
    const std::string trainLabelsKey { "train-labels-idx1-ubyte" };
    const std::string testImagesKey { "t10k-images-idx3-ubyte" };
    const std::string testLabelsKey { "t10k-labels-idx1-ubyte" };

  public:
    void configure(const ConfigurationDescription* configuration);

  protected:
    virtual void configurePolicy(const Eigen::MatrixXd& tmpX, Eigen::MatrixXd& X,
        const Eigen::MatrixXd& tmpY, Eigen::MatrixXd& Y);

  private:
    void imagesLabelsLoad(const std::string& imagesfilename, const std::string& labelsfilename,
        Eigen::MatrixXd& images, Eigen::MatrixXd& labels);
};

#endif /* MNISTDATAFUNCTION_H_ */

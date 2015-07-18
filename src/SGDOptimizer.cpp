/*
 * SGDOptimizer.cpp
 *
 *  Created on: Jul 5, 2015
 *      Author: sam
 */

#include "SGDOptimizer.h"
#include <random>

void SGDOptimizer::optimize(Eigen::VectorXd& theta, DataFunction* dataFunction,
    CostFunction* costFunction)
{
  const int epochs = 3;
  const int minibatch = 250;
  double alpha = 1e-2;
  double momentum = 0.95f;

  double mom = 0.5f;
  int momIncrease = 20;

  Eigen::VectorXd velocity = Eigen::VectorXd::Zero(theta.size());

  int it = 0;
  for (int epoch = 0; epoch < epochs; ++epoch)
  {
    for (int i = 0; i < dataFunction->getTrainingX().rows(); i += minibatch)
    {

      ++it;
      if (it == momIncrease)
      {
        std::cout << "Momentum increased: " << it << std::endl;
        mom = momentum;
      }

      Eigen::MatrixXd mb_data = dataFunction->getTrainingX().block(i, 0, minibatch,
          dataFunction->getTrainingX().cols());
      Eigen::MatrixXd mb_labels = dataFunction->getTrainingY().block(i, 0, minibatch,
          dataFunction->getTrainingY().cols());

      Eigen::VectorXd grad;
      double cost = costFunction->evaluate(theta, mb_data, mb_labels, grad);

      velocity = velocity * mom + alpha * grad;
      theta -= velocity;

      std::cout << "Epoc: " << epoch << " cost: " << cost << " it: " << it << std::endl;

    }
    alpha /= 2.0f;
  }
}


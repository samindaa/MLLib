/*
 * MatrixToLatex.cpp
 *
 *  Created on: Jul 22, 2015
 *      Author: sam
 */

#include "MatrixToLatex.h"
#include <iomanip>

void MatrixToLatex::toLatex(const Matrix_t& M)
{
  std::cout << M << std::endl;

  int nrow = M.rows();
  int ncol = M.cols();

  Vector_t rowsum = M.rowwise().sum();
  Matrix_t RowSum = rowsum.replicate(1, M.cols());
  Matrix_t M2 = M.cwiseQuotient(RowSum) * 100.0f;

  std::cout << "\\begin{table}" << std::endl;
  std::cout << "\\caption{}" << std::endl;
  std::cout << "\\begin{tabular}{";

  std::cout << "|c";
  for (int col = 0; col < ncol; col++)
    std::cout << "|c";
  std::cout << "|}" << std::endl;
  //std::cout << "\%\%\% add table  head here" << std::endl;
  std::cout << "\\hline " << std::endl;
  //for (int col = 0; col < ncol - 1; col++)
  //  std::cout << " &";
  //std::cout << "\t\\\\\n";
  for (int row = 0; row < nrow; row++)
  {
    //std::cout << "\\hline ";
    std::cout << " & ";
    for (int col = 0; col < ncol; col++)
    {
      //std::cout << M2(row, col);
      if (M2(row, col) == 0 || M2(row, col) == 100)
        std::cout << M2(row, col);
      else
        printf("%0.1f", M2(row, col));
      if (col < ncol - 1)
        std::cout << " &  ";
    }
    if (row < nrow - 1)
      std::cout << " \\\\ \\hline" << std::endl;
  }
  std::cout << " \\\\ \\hline" << std::endl;
  std::cout << "\\end{tabular}" << std::endl;
  std::cout << "\\end{table}" << std::endl;
}


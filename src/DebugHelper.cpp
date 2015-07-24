/*
 * DebugHelper.cpp
 *
 *  Created on: Jul 23, 2015
 *      Author: sam
 */

#include "DebugHelper.h"
#include <fstream>

void DebugHelper::writeMatrixTopRows(const Matrix_t& M, const int& topRows,
    const std::string& fname)
{
  std::ofstream ofs(fname.c_str());
  if (ofs.is_open())
  {
    ofs << M.topRows(topRows) << std::endl;
    ofs.flush();
  }
  else
  {
    std::cerr << "DebugHelper => file: " << fname << " cannot open." << std::endl;
  }
}

void DebugHelper::writeMatrixBottomRows(const Matrix_t& M, const int& bottomRows,
    const std::string& fname)
{
  std::ofstream ofs(fname.c_str());
  if (ofs.is_open())
  {
    ofs << M.bottomRows(bottomRows) << std::endl;
    ofs.flush();
  }
  else
  {
    std::cerr << "DebugHelper => file: " << fname << " cannot open." << std::endl;
  }
}


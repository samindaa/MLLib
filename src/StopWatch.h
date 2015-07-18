/*
 * StopWatch.h
 *
 *  Created on: Jul 17, 2015
 *      Author: sam
 */

#ifndef STOPWATCH_H_
#define STOPWATCH_H_

#include <iostream>
#include <chrono>

class StopWatch
{
  private:
    std::chrono::high_resolution_clock::time_point m_start;
    std::chrono::high_resolution_clock::time_point m_end;

  public:
    StopWatch() :
        m_start(std::chrono::high_resolution_clock::now())
    {
    }

    ~StopWatch()
    {
    }

    void start()
    {
      m_start = std::chrono::high_resolution_clock::now();
    }

    void end()
    {
      m_end = std::chrono::high_resolution_clock::now();
    }

    std::chrono::milliseconds ms() const
    {
      return std::chrono::duration_cast<std::chrono::milliseconds>(m_end - m_start);
    }

    std::chrono::microseconds us() const
    {
      return std::chrono::duration_cast<std::chrono::microseconds>(m_end - m_start);
    }

    double ms_count() const
    {
      return ms().count();
    }

    double us_count() const
    {
      return us().count();
    }

};

#endif /* STOPWATCH_H_ */

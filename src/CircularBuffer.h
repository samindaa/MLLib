/*
 * CircularBuffer.h
 *
 *  Created on: Jun 28, 2015
 *      Author: sam
 */

#ifndef CIRCULARBUFFER_H_
#define CIRCULARBUFFER_H_

#include <algorithm>
#include <memory.h>

template<typename T>
class CircularBuffer
{
  private:
    T * buffer;
    size_t size;
    size_t begin;
    size_t end;
    bool wrap;

  public:
    /**
     * create a CircularBuffer with space for up to size elements.
     */
    explicit CircularBuffer(size_t size) :
        size(size), begin(0), end(0), wrap(false)
    {
      buffer = new T[size];
    }

    /**
     * destructor
     */
    ~CircularBuffer()
    {
      delete[] buffer;
    }

    size_t write(const T * data, size_t n)
    {
      n = std::min(n, getFree());

      if (n == 0)
      {
        return n;
      }

      const size_t first_chunk = std::min(n, size - end);
      memcpy(buffer + end, data, first_chunk * sizeof(T));
      end = (end + first_chunk) % size;

      if (first_chunk < n)
      {
        const size_t second_chunk = n - first_chunk;
        memcpy(buffer + end, data + first_chunk, second_chunk * sizeof(T));
        end = (end + second_chunk) % size;
      }

      if (begin == end)
      {
        wrap = true;
      }

      return n;
    }

    size_t read(T * dest, size_t n)
    {
      n = std::min(n, getOccupied());

      if (n == 0)
      {
        return n;
      }

      if (wrap)
      {
        wrap = false;
      }

      const size_t first_chunk = std::min(n, size - begin);
      memcpy(dest, buffer + begin, first_chunk * sizeof(T));
      begin = (begin + first_chunk) % size;

      if (first_chunk < n)
      {
        const size_t second_chunk = n - first_chunk;
        memcpy(dest + first_chunk, buffer + begin, second_chunk * sizeof(T));
        begin = (begin + second_chunk) % size;
      }
      return n;
    }

    size_t getOccupied()
    {
      if (end == begin)
      {
        return wrap ? size : 0;
      }
      else if (end > begin)
      {
        return end - begin;
      }
      else
      {
        return size + end - begin;
      }
    }

    size_t getFree()
    {
      return size - getOccupied();
    }

};

#endif /* CIRCULARBUFFER_H_ */

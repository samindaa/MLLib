/*
 * AsoundLib.cpp
 *
 *  Created on: Jun 21, 2015
 *      Author: sam
 */

#include "AsoundLib.h"

AsoundLib::AsoundLib(const unsigned int& rate, const int& channels, const snd_pcm_uframes_t& period,
    const snd_pcm_format_t& format) :
    rate(rate), channels(channels), period(period), pin(nullptr), pout(nullptr), hw(nullptr), //
    sw(nullptr), boundary(0), frames(0), buf_size(0), format(format)
{
}

AsoundLib::~AsoundLib()
{
  if (pout)
    snd_pcm_close(pout);
  if (pin)
    snd_pcm_close(pin);
}

void AsoundLib::configure()
{
  CHECK(snd_pcm_open(&pin, "default", SND_PCM_STREAM_CAPTURE, 0));
  CHECK(snd_pcm_open(&pout, "default", SND_PCM_STREAM_PLAYBACK, 0));

  snd_pcm_hw_params_alloca(&hw);
  CHECK(snd_pcm_hw_params_any(pin, hw));
  CHECK(snd_pcm_hw_params_set_access(pin, hw, SND_PCM_ACCESS_RW_INTERLEAVED));
  CHECK(snd_pcm_hw_params_set_format(pin, hw, SND_PCM_FORMAT_S16_LE));
  CHECK(snd_pcm_hw_params_set_channels(pin, hw, channels));
  CHECK(snd_pcm_hw_params_set_rate(pin, hw, rate, 0));
  CHECK(snd_pcm_hw_params_set_period_size(pin, hw, period, 0));
  //CHECK(snd_pcm_hw_params_set_buffer_size(pin, hw, 16 * PERIOD));
  CHECK(snd_pcm_hw_params(pin, hw));

  CHECK(snd_pcm_hw_params_any(pout, hw));
  CHECK(snd_pcm_hw_params_set_access(pout, hw, SND_PCM_ACCESS_RW_INTERLEAVED));
  CHECK(snd_pcm_hw_params_set_format(pout, hw, SND_PCM_FORMAT_S16_LE));
  CHECK(snd_pcm_hw_params_set_channels(pout, hw, channels));
  CHECK(snd_pcm_hw_params_set_rate_near(pout, hw, &rate, 0));
  CHECK(snd_pcm_hw_params_set_period_size(pout, hw, period, 0));
//  CHECK(snd_pcm_hw_params_set_buffer_size(pout, hw, 2 * PERIOD));
  CHECK(snd_pcm_hw_params(pout, hw));

  snd_pcm_sw_params_alloca(&sw);
  CHECK(snd_pcm_sw_params_current(pin, sw));
  CHECK(snd_pcm_sw_params_get_boundary(sw, &boundary));
  CHECK(snd_pcm_sw_params_set_start_threshold(pin, sw, boundary));
  CHECK(snd_pcm_sw_params(pin, sw));

  CHECK(snd_pcm_sw_params_current(pout, sw));
  CHECK(snd_pcm_sw_params_get_boundary(sw, &boundary));
  CHECK(snd_pcm_sw_params_set_avail_min(pout, sw, period));
  CHECK(snd_pcm_sw_params_set_start_threshold(pout, sw, boundary));
  CHECK(snd_pcm_sw_params(pout, sw));

  //CHECK(snd_pcm_link(pout, pin));

  // stats

  /* Resume information */
  std::cout << "PCM name: " << snd_pcm_name(pout) << std::endl;
  std::cout << "PCM state: " << snd_pcm_state_name(snd_pcm_state(pout)) << std::endl;

  unsigned int tmp;
  snd_pcm_hw_params_get_channels(hw, &tmp);
  std::cout << "channels: " << tmp << std::endl;
  if (tmp == 1)
    std::cout << "(mono)" << std::endl;
  else if (tmp == 2)
    std::cout << "(stereo)" << std::endl;

  snd_pcm_hw_params_get_rate(hw, &tmp, 0);
  printf("rate: %d bps\n", tmp);

  /* Allocate buffer to hold single period */
  snd_pcm_hw_params_get_period_size(hw, &frames, 0);
  buf_size = frames * channels;
  std::cout << "buf_size: " << buf_size << std::endl;

}

void AsoundLib::playback(const short* buf, const size_t& bufLen)
{
  CHECK(snd_pcm_prepare(pout));
  size_t start_ptr = 0;
  while (start_ptr < bufLen)
  {
    if (start_ptr + buf_size > bufLen)
    {
      // With padding
      size_t current_buff_size = bufLen - start_ptr;
      std::cout << "Error: start_ptr: " << start_ptr << " end_ptr: "
          << (start_ptr + current_buff_size) << " bufLen:" << bufLen << std::endl;
      exit(EXIT_FAILURE);
    }

    //std::cout << "start_ptr: " << start_ptr << " end_ptr: " << (start_ptr + buf_size) << std::endl;
    CHECK(snd_pcm_writei(pout, buf + start_ptr, buf_size));
    start_ptr += buf_size;
  }

  assert(start_ptr == bufLen);
  snd_pcm_drain(pout);
}

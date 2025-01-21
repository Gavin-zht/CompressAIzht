/* Copyright (c) 2021-2024, InterDigital Communications, Inc
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted (subject to the limitations in the disclaimer
 * below) provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice,
 *   this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of InterDigital Communications, Inc nor the names of its
 *   contributors may be used to endorse or promote products derived from this
 *   software without specific prior written permission.
 *
 * NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
 * THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
 * NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
 * PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
 * OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
 * ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "rans_interface.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <torch/extension.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "rans64.h"

namespace py = pybind11;

/* probability range, this could be a parameter... */
constexpr int precision = 16;

constexpr uint16_t bypass_precision = 4; /* number of bits in bypass mode */
constexpr uint16_t max_bypass_val = (1 << bypass_precision) - 1;

namespace {

/* We only run this in debug mode as its costly... */
void assert_cdfs(const std::vector<std::vector<int>> &cdfs,
                 const std::vector<int> &cdfs_sizes) {
  for (int i = 0; i < static_cast<int>(cdfs.size()); ++i) {
    assert(cdfs[i][0] == 0);
    assert(cdfs[i][cdfs_sizes[i] - 1] == (1 << precision));
    for (int j = 0; j < cdfs_sizes[i] - 1; ++j) {
      assert(cdfs[i][j + 1] > cdfs[i][j]);
    }
  }
}

// std::vector<std::vector<int32_t>> make_cdfs_vector_from_tensor(
//   const torch::Tensor &cdfs, const std::vector<int32_t> &cdfs_sizes) {
//   assert(cdfs.dim() == 2);
//   assert(cdfs.size(0) == cdfs_sizes.size());
//   assert(cdfs.dtype() == torch::kInt32);

//   auto num_samples = cdfs.size(1);
//   auto *ptr = reinterpret_cast<int32_t*>(cdfs.data_ptr());

//   std::vector<std::vector<int32_t>> result;

//   for (auto cdf_size : cdfs_sizes) {
//     std::vector<int32_t> cdf_vec(ptr, ptr + cdf_size);
//     ptr += num_samples;
//     result.push_back(std::move(cdf_vec));
//   }

//   return result;
// }

/* Support only 16 bits word max */
inline void Rans64EncPutBits(Rans64State *r, uint32_t **pptr, uint32_t val,
                             uint32_t nbits) {
  assert(nbits <= 16);
  assert(val < (1u << nbits));

  /* Re-normalize */
  uint64_t x = *r;
  uint32_t freq = 1 << (16 - nbits);
  uint64_t x_max = ((RANS64_L >> 16) << 32) * freq;
  if (x >= x_max) {
    *pptr -= 1;
    **pptr = (uint32_t)x;
    x >>= 32;
    Rans64Assert(x < x_max);
  }

  /* x = C(s, x) */
  *r = (x << nbits) | val;
}

inline uint32_t Rans64DecGetBits(Rans64State *r, uint32_t **pptr,
                                 uint32_t n_bits) {
  uint64_t x = *r;
  uint32_t val = x & ((1u << n_bits) - 1);

  /* Re-normalize */
  x = x >> n_bits;
  if (x < RANS64_L) {
    x = (x << 32) | **pptr;
    *pptr += 1;
    Rans64Assert(x >= RANS64_L);
  }

  *r = x;

  return val;
}
} // namespace
/*
* 功能： 这个函数 BufferedRansEncoder::encode_with_indexes 是一个用于将符号序列编码为RANS（Range ANS）编码数据的函数。这个函数是RANS编码器的一部分，用于将符号序列编码为一个紧凑的比特流。
* RANS是一种用于无损数据压缩的熵编码算法，它结合了算术编码的效率和Huffman编码的简单性。

* 参数: symbols：要编码的符号序列, indexes：每个符号对应的CDF（累积分布函数）索引, cdfs：所有符号的CDF列表, cdfs_sizes：每个CDF的大小, offsets：每个CDF的偏移量。

* 总体逻辑: 这个函数 BufferedRansEncoder::encode_with_indexes 实现了将符号序列编码为RANS编码数据的过程。它通过反向遍历符号序列，使用每个符号的CDF进行编码，并处理超出CDF范围的值。
* 编码结果存储在 _syms 缓冲区中，最终可以被进一步处理并输出为紧凑的比特流。

*/
void BufferedRansEncoder::encode_with_indexes(
    const std::vector<int32_t> &symbols, const std::vector<int32_t> &indexes,
    const std::vector<std::vector<int32_t>> &cdfs,
    const std::vector<int32_t> &cdfs_sizes,
    const std::vector<int32_t> &offsets) 
{
  assert(cdfs.size() == cdfs_sizes.size()); //* 确保 cdfs 和 cdfs_sizes 的大小相同。
  assert_cdfs(cdfs, cdfs_sizes);

  // backward loop on symbols from the end;
  for (size_t i = 0; i < symbols.size(); ++i) { 
    //* 从符号序列的末尾开始遍历，这是因为RANS编码是反向进行的，即先编码最后一个符号，再编码倒数第二个符号，依此类推。
    //* 假设当前遍历到了 第i个符号
    const int32_t cdf_idx = indexes[i]; //* 对于第i个符号，获取其对应的CDF索引 cdf_idx。
    assert(cdf_idx >= 0);
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];  //* 获取第i个符号对应的CDF cdf 
    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;  //* 获取第i个符号对应的最大值 max_value，max_value 是 cdfs_sizes[cdf_idx] - 2
    
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    int32_t value = symbols[i] - offsets[cdf_idx];  //* 计算符号的实际值 value，value = symbols[i] - offsets[cdf_idx]
    //符号值symbols可能在CDF的下标之外，有可能找不到对应的CDF，因此要通过offset确保符号值能够正确地映射到CDF中的某个概率区间，所以value是实际对应到单个CDF中的下标
    //cdfs[cdf_idx][value]是要取出的CDF区间的起始点
    uint32_t raw_val = 0;
    //符号值过大或为负时，使用一下if——else会控制进入旁路编码模式
    if (value < 0) {
      raw_val = -2 * value - 1;
      value = max_value;
    } else if (value >= max_value) {
      raw_val = 2 * (value - max_value);
      value = max_value;
    }

    assert(value >= 0);
    assert(value < cdfs_sizes[cdf_idx] - 1);
    //断言确保value在cdf下标范围内

    _syms.push_back({static_cast<uint16_t>(cdf[value]),
                     static_cast<uint16_t>(cdf[value + 1] - cdf[value]),
                     false});
    //CDF是累积分布函数数组，其中每个元素表示到当前符号为止的累积概率，注意是累加的值，因此cdf[value + 1] - cdf[value]的差值才是我们需要的概率范围range
    //也就是说每个符号的概率要通过相邻两个CDF值之间的差来计算，实际上就代表了当前符号的概率


    /* 
    Bypass coding mode (value == max_value -> sentinel flag) 
    * 旁路编码模式用于处理超出CDF范围的值。
    * 在这种模式下，编码器使用固定的概率模型（通常是均匀分布）来编码值，而不是使用CDF。这样可以处理任意大的值，而不会使CDF变得过于复杂。    


    */
    if (value == max_value) {
      /* Determine the number of bypasses (in bypass_precision size) needed to
       * encode the raw value. */
      int32_t n_bypass = 0;
      while ((raw_val >> (n_bypass * bypass_precision)) != 0) {
        ++n_bypass;
      }
      //一直右移 raw_val，用其位数确定旁路的执行次数

      /* Encode number of bypasses */
      int32_t val = n_bypass;
      while (val >= max_bypass_val) {
        _syms.push_back({max_bypass_val, max_bypass_val + 1, true});
        val -= max_bypass_val;
      }
      _syms.push_back(
          {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});

      /* Encode raw value */
      for (int32_t j = 0; j < n_bypass; ++j) {
        const int32_t val =
            (raw_val >> (j * bypass_precision)) & max_bypass_val;//raw_val 右移 j * bypass_precision 位后取出作为起始点
        _syms.push_back(
            {static_cast<uint16_t>(val), static_cast<uint16_t>(val + 1), true});
      }
    }
  }
}

// void BufferedRansEncoder::encode_with_indexes(
//     const std::vector<int32_t> &symbols, const std::vector<int32_t> &indexes,
//     const torch::Tensor &cdfs,
//     const std::vector<int32_t> &cdfs_sizes,
//     const std::vector<int32_t> &offsets) {
//   return encode_with_indexes(symbols, indexes,
//                              make_cdfs_vector_from_tensor(cdfs, cdfs_sizes),
//                              cdfs_sizes, offsets);
// }

py::bytes BufferedRansEncoder::flush() {
  Rans64State rans;
  Rans64EncInit(&rans);

  std::vector<uint32_t> output(_syms.size(), 0xCC); // too much space ?
  uint32_t *ptr = output.data() + output.size();
  assert(ptr != nullptr);

  while (!_syms.empty()) {
    const RansSymbol sym = _syms.back();

    if (!sym.bypass) {
      Rans64EncPut(&rans, &ptr, sym.start, sym.range, precision);
    } else {
      // unlikely...
      Rans64EncPutBits(&rans, &ptr, sym.start, bypass_precision);
    }
    _syms.pop_back();
  }

  Rans64EncFlush(&rans, &ptr);

  const int nbytes =
      std::distance(ptr, output.data() + output.size()) * sizeof(uint32_t);
  return std::string(reinterpret_cast<char *>(ptr), nbytes);
}

py::bytes
RansEncoder::encode_with_indexes(const std::vector<int32_t> &symbols,
                                 const std::vector<int32_t> &indexes,
                                 const std::vector<std::vector<int32_t>> &cdfs,
                                 const std::vector<int32_t> &cdfs_sizes,
                                 const std::vector<int32_t> &offsets) {

  BufferedRansEncoder buffered_rans_enc;
  buffered_rans_enc.encode_with_indexes(symbols, indexes, cdfs, cdfs_sizes,
                                        offsets);
  return buffered_rans_enc.flush();
}

// py::bytes
// RansEncoder::encode_with_indexes(const std::vector<int32_t> &symbols,
//                                  const std::vector<int32_t> &indexes,
//                                  const torch::Tensor &cdfs,
//                                  const std::vector<int32_t> &cdfs_sizes,
//                                  const std::vector<int32_t> &offsets) {
//   return encode_with_indexes(symbols, indexes,
//                              make_cdfs_vector_from_tensor(cdfs, cdfs_sizes),
//                              cdfs_sizes, offsets);
// }

std::vector<int32_t>
RansDecoder::decode_with_indexes(const std::string &encoded,
                                 const std::vector<int32_t> &indexes,
                                 const std::vector<std::vector<int32_t>> &cdfs,
                                 const std::vector<int32_t> &cdfs_sizes,
                                 const std::vector<int32_t> &offsets) {
  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  std::vector<int32_t> output(indexes.size());

  Rans64State rans;
  uint32_t *ptr = (uint32_t *)encoded.data();
  assert(ptr != nullptr);
  Rans64DecInit(&rans, &ptr);

  for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    const int32_t offset = offsets[cdf_idx];

    const uint32_t cum_freq = Rans64DecGet(&rans, precision);

    const auto cdf_end = cdf.begin() + cdfs_sizes[cdf_idx];
    const auto it = std::find_if(cdf.begin(), cdf_end,
                                 [cum_freq](int v) { return v > cum_freq; });
    assert(it != cdf_end + 1);
    const uint32_t s = std::distance(cdf.begin(), it) - 1;

    Rans64DecAdvance(&rans, &ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
      /* Bypass decoding mode */
      int32_t val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
      int32_t n_bypass = val;

      while (val == max_bypass_val) {
        val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&rans, &ptr, bypass_precision);
        assert(val <= max_bypass_val);
        raw_val |= val << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    output[i] = value + offset;
  }

  return output;
}

// std::vector<int32_t>
// RansDecoder::decode_with_indexes(const std::string &encoded,
//                                  const std::vector<int32_t> &indexes,
//                                  const torch::Tensor &cdfs,
//                                  const std::vector<int32_t> &cdfs_sizes,
//                                  const std::vector<int32_t> &offsets) {
//   return decode_with_indexes(encoded, indexes,
//                              make_cdfs_vector_from_tensor(cdfs, cdfs_sizes),
//                              cdfs_sizes, offsets);
// }

void RansDecoder::set_stream(const std::string &encoded) {
  _stream = encoded;
  uint32_t *ptr = (uint32_t *)_stream.data();
  assert(ptr != nullptr);
  _ptr = ptr;
  Rans64DecInit(&_rans, &_ptr);
}

std::vector<int32_t>
RansDecoder::decode_stream(const std::vector<int32_t> &indexes,
                           const std::vector<std::vector<int32_t>> &cdfs,
                           const std::vector<int32_t> &cdfs_sizes,
                           const std::vector<int32_t> &offsets) {
  assert(cdfs.size() == cdfs_sizes.size());
  assert_cdfs(cdfs, cdfs_sizes);

  std::vector<int32_t> output(indexes.size());

  assert(_ptr != nullptr);

  for (int i = 0; i < static_cast<int>(indexes.size()); ++i) {
    const int32_t cdf_idx = indexes[i];
    assert(cdf_idx >= 0);
    assert(cdf_idx < cdfs.size());

    const auto &cdf = cdfs[cdf_idx];

    const int32_t max_value = cdfs_sizes[cdf_idx] - 2;
    assert(max_value >= 0);
    assert((max_value + 1) < cdf.size());

    const int32_t offset = offsets[cdf_idx];

    const uint32_t cum_freq = Rans64DecGet(&_rans, precision);

    const auto cdf_end = cdf.begin() + cdfs_sizes[cdf_idx];
    const auto it = std::find_if(cdf.begin(), cdf_end,
                                 [cum_freq](int v) { return v > cum_freq; });
    assert(it != cdf_end + 1);
    const uint32_t s = std::distance(cdf.begin(), it) - 1;

    Rans64DecAdvance(&_rans, &_ptr, cdf[s], cdf[s + 1] - cdf[s], precision);

    int32_t value = static_cast<int32_t>(s);

    if (value == max_value) {
      /* Bypass decoding mode */
      int32_t val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
      int32_t n_bypass = val;

      while (val == max_bypass_val) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        n_bypass += val;
      }

      int32_t raw_val = 0;
      for (int j = 0; j < n_bypass; ++j) {
        val = Rans64DecGetBits(&_rans, &_ptr, bypass_precision);
        assert(val <= max_bypass_val);
        raw_val |= val << (j * bypass_precision);
      }
      value = raw_val >> 1;
      if (raw_val & 1) {
        value = -value - 1;
      } else {
        value += max_value;
      }
    }

    output[i] = value + offset;
  }

  return output;
}

// std::vector<int32_t>
// RansDecoder::decode_stream(const std::vector<int32_t> &indexes,
//                            const torch::Tensor &cdfs,
//                            const std::vector<int32_t> &cdfs_sizes,
//                            const std::vector<int32_t> &offsets) {
//   return decode_stream(indexes, make_cdfs_vector_from_tensor(cdfs,
//   cdfs_sizes),
//                        cdfs_sizes, offsets);
// }

PYBIND11_MODULE(ans, m) {
  //这个部分将C++代码做成一个库绑定到Python
  m.attr("__name__") = "compressai.ans";

  m.doc() = "range Asymmetric Numeral System python bindings";

  py::class_<BufferedRansEncoder>(m, "BufferedRansEncoder")
      .def(py::init<>())
      .def("encode_with_indexes",
           py::overload_cast<
               const std::vector<int32_t> &, const std::vector<int32_t> &,
               const std::vector<std::vector<int32_t>> &,
               const std::vector<int32_t> &, const std::vector<int32_t> &>(
               &BufferedRansEncoder::encode_with_indexes))
      // .def("encode_with_indexes",
      //      py::overload_cast<
      //          const std::vector<int32_t> &,
      //          const std::vector<int32_t> &,
      //          const torch::Tensor &,
      //          const std::vector<int32_t> &,
      //          const std::vector<int32_t> &
      //          >(&BufferedRansEncoder::encode_with_indexes))
      .def("flush", &BufferedRansEncoder::flush);

  py::class_<RansEncoder>(m, "RansEncoder")
      .def(py::init<>())
      .def("encode_with_indexes",
           py::overload_cast<
               const std::vector<int32_t> &, const std::vector<int32_t> &,
               const std::vector<std::vector<int32_t>> &,
               const std::vector<int32_t> &, const std::vector<int32_t> &>(
               &RansEncoder::encode_with_indexes));
  // .def("encode_with_indexes",
  //      py::overload_cast<
  //          const std::vector<int32_t> &,
  //          const std::vector<int32_t> &,
  //          const torch::Tensor &,
  //          const std::vector<int32_t> &,
  //          const std::vector<int32_t> &
  //          >(&RansEncoder::encode_with_indexes));

  py::class_<RansDecoder>(m, "RansDecoder")
      .def(py::init<>())
      .def("set_stream", &RansDecoder::set_stream)
      .def("decode_stream",
           py::overload_cast<const std::vector<int32_t> &,
                             const std::vector<std::vector<int32_t>> &,
                             const std::vector<int32_t> &,
                             const std::vector<int32_t> &>(
               &RansDecoder::decode_stream))
      // .def("decode_stream",
      //      py::overload_cast<
      //          const std::vector<int32_t> &,
      //          const torch::Tensor &,
      //          const std::vector<int32_t> &,
      //          const std::vector<int32_t> &
      //          >(&RansDecoder::decode_stream))
      .def("decode_with_indexes",
           py::overload_cast<const std::string &, const std::vector<int32_t> &,
                             const std::vector<std::vector<int32_t>> &,
                             const std::vector<int32_t> &,
                             const std::vector<int32_t> &>(
               &RansDecoder::decode_with_indexes),
           "Decode a string to a list of symbols");
  // .def("decode_with_indexes",
  //      py::overload_cast<
  //          const std::string &,
  //          const std::vector<int32_t> &,
  //          const torch::Tensor &,
  //          const std::vector<int32_t> &,
  //          const std::vector<int32_t> &
  //          >(&RansDecoder::decode_with_indexes),
  //      "Decode a string to a list of symbols");
}

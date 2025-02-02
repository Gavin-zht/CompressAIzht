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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

/*
*功能
* pmf_to_quantized_cdf 函数用于将概率质量函数（PMF）转换为累积分布函数（CDF）。这是熵编码中的一个关键步骤，用于将概率分布转换为适合熵编码的格式。

*参数
* pmf：概率质量函数，类型为 std::vector<float>，表示每个符号的概率， pmf向量中存放的最后一个数字为tail
* precision：精度，类型为 int，表示累积分布函数的精度。

*返回值
* 返回值是一个 std::vector<uint32_t>，表示量化后的累积分布函数（CDF）, cdf的维度为pmf的长度+1, 并且cdf[0]=0, cdf[cdf_length]=1<<precision

*整体执行逻辑
* 检查 pmf 中的每个概率值是否有效（非负且有限）。
* 创建一个大小为 pmf.size() + 1 的零向量 cdf，并初始化第一个元素为 0。
* 将 pmf 中的概率值转换为频率，并存储在 cdf 中。
* 计算总频率，并将其归一化到 [0, 1 << precision] 范围内。
* 计算累积和，确保 cdf 的最后一个元素等于 1 << precision。
* 如果存在频率为零的符号，尝试从低频率符号中“窃取”频率，以确保所有符号都有非零频率。
* 返回量化后的 cdf。


*/
std::vector<uint32_t> pmf_to_quantized_cdf(const std::vector<float> &pmf,
                                           int precision) {
  /* NOTE(begaintj): ported from `ryg_rans` public implementation. Not optimal
   * although it's only run once per model after training. See TF/compression
   * implementation for an optimized version. */

//* 以下代码用于遍历 pmf 中的每个概率值 p，检查其是否小于 0 或不是有限值。如果发现无效值，抛出异常。
  for (float p : pmf) {
    if (p < 0 || !std::isfinite(p)) {
      throw std::domain_error(
          std::string("Invalid `pmf`, non-finite or negative element found: ") +
          std::to_string(p));
    }
  }
//* 以上代码用于遍历 pmf 中的每个概率值 p，检查其是否小于 0 或不是有限值。如果发现无效值，抛出异常。  

  std::vector<uint32_t> cdf(pmf.size() + 1);  //* 创建一个大小为 pmf.size() + 1 的零向量 cdf，并初始化第一个元素为 0
  cdf[0] = 0; /* freq 0 */

  std::transform(pmf.begin(), pmf.end(), cdf.begin() + 1,
                 [=](float p) { return std::round(p * (1 << precision)); }); //* [=](float p) { return std::round(p * (1 << precision)); 是一个 lambda 表达式，用于将每个概率值 p 转换为频率。具体来说，它将概率值乘以 2^precision，然后四舍五入到最近的整数  
  //* 使用 std::transform 将 pmf 中的概率值转换为频率，并存储在 cdf 中。频率是通过将概率值乘以 1 << precision 并四舍五入得到的               
  //* 具体执行过程：
  //* 遍历 pmf 数组：
  //* std::transform 从 pmf.begin() 开始，到 pmf.end() 结束，逐个处理 pmf 中的每个元素。
  //* 应用 lambda 表达式：
  //* 对于 pmf 中的每个概率值 p，lambda 表达式将其乘以 2^precision，然后使用 std::round 函数四舍五入到最近的整数。1 << precision 是位移操作，等同于 2^precision，用于将概率值缩放到指定的精度范围。
  //* 存储结果：
  //* 转换后的频率值被存储在 cdf 数组中，从 cdf.begin() + 1 开始
  
  //* 举例
  // 假设 pmf 和 precision 如下：
  // std::vector<float> pmf = {0.1, 0.2, 0.3, 0.4};
  // int precision = 8;
  // precision = 8 表示我们将概率值缩放到 2^8 = 256 的范围。
  // 对于 pmf 中的每个概率值：
  // 0.1 * 256 = 25.6，四舍五入后为 26。
  // 0.2 * 256 = 51.2，四舍五入后为 51。
  // 0.3 * 256 = 76.8，四舍五入后为 77。
  // 0.4 * 256 = 102.4，四舍五入后为 102。


  const uint32_t total = std::accumulate(cdf.begin(), cdf.end(), 0);  //* 使用 std::accumulate 计算 cdf 中的所有频率之和。
  if (total == 0) { //* 如果总频率为 0，抛出异常
    throw std::domain_error("Invalid `pmf`: at least one element must have a "
                            "non-zero probability.");
  }

  std::transform(cdf.begin(), cdf.end(), cdf.begin(),
                 [precision, total](uint32_t p) {
                   return ((static_cast<uint64_t>(1 << precision) * p) / total);
                 });  
  //* 将 cdf 中的频率归一化到 [0, 1 << precision] 范围内
  //* 具体执行过程
  // 遍历 cdf 数组：
  // std::transform 从 cdf.begin() 开始，到 cdf.end() 结束，逐个处理 cdf 中的每个元素。
  // 应用 lambda 表达式：
  // 对于 cdf 中的每个频率值 p，lambda 表达式将其归一化到 [0, 1 << precision] 范围内。具体来说，它将频率值 p 乘以 2^precision，然后除以总频率 total。1 << precision 是位移操作，等同于 2^precision，用于将频率值缩放到指定的精度范围。
  // static_cast<uint64_t> 用于确保在乘法操作中不会发生溢出。
  // 存储结果：
  // 转换后的频率值被存储在 cdf 数组中，从 cdf.begin() 开始  

  std::partial_sum(cdf.begin(), cdf.end(), cdf.begin());  //* 使用 std::partial_sum 计算 cdf 的累积和，cdf[i] = cdf[0]+cdf[1]+...cdf[i]
  cdf.back() = 1 << precision;  //* 使用赋值，确保最后一个元素等于 1 << precision

//* 至此，cdf[i]就表示前i个位置的累计频率 
  for (int i = 0; i < static_cast<int>(cdf.size() - 1); ++i) {
    //* 遍历CDF中的每个取值
    if (cdf[i] == cdf[i + 1]) { //* 如果cdf[i+1]==cdf[i]， 就表明pmf[i+1] ==0, 表示pmf的第i+1个位置的概率为0
      //* 如果发现某个符号的频率为零，尝试从低频率符号中“窃取”频率，以确保所有符号都有非零频率
      /* Try to steal frequency from low-frequency symbols */
      uint32_t best_freq = ~0u;
      int best_steal = -1;
      for (int j = 0; j < static_cast<int>(cdf.size()) - 1; ++j) {
        uint32_t freq = cdf[j + 1] - cdf[j];
        if (freq > 1 && freq < best_freq) {
          best_freq = freq;
          best_steal = j;
        }
      }

      assert(best_steal != -1);

      if (best_steal < i) {
        for (int j = best_steal + 1; j <= i; ++j) {
          cdf[j]--;
        }
      } else {
        assert(best_steal > i);
        for (int j = i + 1; j <= best_steal; ++j) {
          cdf[j]++;
        }
      }
    }
  }

//* 验证 cdf 的第一个元素为 0，最后一个元素为 1 << precision，并且每个元素都小于其后续元素
  assert(cdf[0] == 0);
  assert(cdf.back() == (1 << precision));
  for (int i = 0; i < static_cast<int>(cdf.size()) - 1; ++i) {
    assert(cdf[i + 1] > cdf[i]);
  }

  return cdf;
}

PYBIND11_MODULE(_CXX, m) {
  m.attr("__name__") = "compressai._CXX";

  m.doc() = "C++ utils";

  m.def("pmf_to_quantized_cdf", &pmf_to_quantized_cdf,
        "Return quantized CDF for a given PMF");
}

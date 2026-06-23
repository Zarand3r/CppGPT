// cppgpt device tag: the seam that lets the planned CUDA phase add kernels
// behind the same op signatures without changing callers. v1 is CPU-only; every
// op carries a Device and asserts CPU at entry (PLAN invariant 8). DType is
// intentionally omitted until a second dtype exists — v1 is fp32 throughout.
#pragma once

namespace cppgpt {

enum class Device { CPU };

}  // namespace cppgpt

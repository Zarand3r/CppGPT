// tensor.h
#pragma once

#include <vector>
#include <memory>
#include <cassert>
#include <iostream>
#include <functional>
#include <stack>
#include <cmath>

namespace cppgpt {
namespace autograd {
    class Function;
}

namespace datatypes {

class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape;
    std::vector<float> grad;
    bool requires_grad;
    std::shared_ptr<autograd::Function> grad_fn = nullptr;

    Tensor(std::vector<float> data, std::vector<int> shape, bool requires_grad = false)
        : data(std::move(data)), shape(std::move(shape)), requires_grad(requires_grad) {
        if (requires_grad) {
            grad.resize(this->data.size(), 0.0f);
        }
    }

    int numel() const {
        int n = 1;
        for (int dim : shape) n *= dim;
        return n;
    }

    void zero_grad() {
        if (requires_grad) std::fill(grad.begin(), grad.end(), 0.0f);
    }

    void backward();

    Tensor add(const Tensor& other);
    Tensor matmul(const Tensor& other);
    Tensor relu();
    Tensor sum();
};

} // namespace datatypes
} // namespace cppgpt
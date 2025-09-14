// autograd.h
#pragma once
#include "cppgpt/datatypes/tensor.h"

namespace cppgpt {
namespace autograd {

class Function {
public:
    std::vector<datatypes::Tensor*> inputs;
    virtual ~Function() = default;
    virtual void backward(datatypes::Tensor& output_grad) = 0;
};

class AddFunction : public Function {
public:
    void backward(datatypes::Tensor& output_grad) override {
        auto a = inputs[0];
        auto b = inputs[1];
        if (a->requires_grad) {
            for (size_t i = 0; i < a->grad.size(); ++i) a->grad[i] += output_grad.grad[i];
        }
        if (b->requires_grad) {
            for (size_t i = 0; i < b->grad.size(); ++i) b->grad[i] += output_grad.grad[i];
        }
    }
};

class MatMulFunction : public Function {
public:
    void backward(datatypes::Tensor& output_grad) override {
        auto a = inputs[0];
        auto b = inputs[1];

        int m = a->shape[0];
        int k = a->shape[1];
        int n = b->shape[1];

        if (a->requires_grad) {
            for (int i = 0; i < m; ++i)
                for (int j = 0; j < k; ++j)
                    for (int l = 0; l < n; ++l)
                        a->grad[i * k + j] += output_grad.grad[i * n + l] * b->data[j * n + l];
        }
        if (b->requires_grad) {
            for (int i = 0; i < k; ++i)
                for (int j = 0; j < n; ++j)
                    for (int l = 0; l < m; ++l)
                        b->grad[i * n + j] += a->data[l * k + i] * output_grad.grad[l * n + j];
        }
    }
};

class ReLUFunction : public Function {
public:
    void backward(datatypes::Tensor& output_grad) override {
        auto x = inputs[0];
        for (size_t i = 0; i < x->data.size(); ++i) {
            float grad = x->data[i] > 0 ? 1.0f : 0.0f;
            if (x->requires_grad)
                x->grad[i] += grad * output_grad.grad[i];
        }
    }
};

class SumFunction : public Function {
public:
    void backward(datatypes::Tensor& output_grad) override {
        auto x = inputs[0];
        if (x->requires_grad) {
            for (auto& g : x->grad) g += 1.0f * output_grad.grad[0];
        }
    }
};

} // namespace autograd
} // namespace cppgpt
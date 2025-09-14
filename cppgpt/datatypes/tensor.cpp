// tensor.cpp
#include "cppgpt/datatypes/tensor.h"
#include "cppgpt/autograd/autograd.h"

namespace cppgpt {
namespace datatypes {

Tensor Tensor::add(const Tensor& other) {
    assert(shape == other.shape);
    std::vector<float> out_data(data.size());
    for (size_t i = 0; i < data.size(); ++i)
        out_data[i] = data[i] + other.data[i];

    Tensor out(out_data, shape, requires_grad || other.requires_grad);
    if (out.requires_grad) {
        auto fn = std::make_shared<autograd::AddFunction>();
        fn->inputs = {const_cast<Tensor*>(this), const_cast<Tensor*>(&other)};
        out.grad_fn = fn;
    }
    return out;
}

Tensor Tensor::matmul(const Tensor& other) {
    assert(shape.size() == 2 && other.shape.size() == 2);
    int m = shape[0], k = shape[1], n = other.shape[1];
    assert(k == other.shape[0]);

    std::vector<float> out_data(m * n, 0.0f);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            for (int l = 0; l < k; ++l)
                out_data[i * n + j] += data[i * k + l] * other.data[l * n + j];

    Tensor out(out_data, {m, n}, requires_grad || other.requires_grad);
    if (out.requires_grad) {
        auto fn = std::make_shared<autograd::MatMulFunction>();
        fn->inputs = {const_cast<Tensor*>(this), const_cast<Tensor*>(&other)};
        out.grad_fn = fn;
    }
    return out;
}

Tensor Tensor::relu() {
    std::vector<float> out_data(data.size());
    for (size_t i = 0; i < data.size(); ++i)
        out_data[i] = std::max(0.0f, data[i]);

    Tensor out(out_data, shape, requires_grad);
    if (out.requires_grad) {
        auto fn = std::make_shared<autograd::ReLUFunction>();
        fn->inputs = {const_cast<Tensor*>(this)};
        out.grad_fn = fn;
    }
    return out;
}

Tensor Tensor::sum() {
    float total = 0.0f;
    for (float v : data) total += v;
    Tensor out({total}, {1}, requires_grad);
    if (requires_grad) {
        auto fn = std::make_shared<autograd::SumFunction>();
        fn->inputs = {const_cast<Tensor*>(this)};
        out.grad_fn = fn;
    }
    return out;
}

void Tensor::backward() {
    grad = std::vector<float>(data.size(), 1.0f);
    std::stack<Tensor*> stack;
    stack.push(this);
    while (!stack.empty()) {
        Tensor* t = stack.top(); stack.pop();
        if (t->grad_fn) {
            t->grad_fn->backward(*t);
            for (auto& input : t->grad_fn->inputs)
                if (input->requires_grad)
                    stack.push(input);
        }
    }
}

} // namespace datatypes
} // namespace cppgpt

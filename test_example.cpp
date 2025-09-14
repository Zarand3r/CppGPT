#include "cppgpt/datatypes/tensor.h"
#include <iostream>

int main() {
    using namespace cppgpt::datatypes;
    
    // Create tensors with requires_grad=true
    Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}, true);
    Tensor b({5.0f, 6.0f, 7.0f, 8.0f}, {2, 2}, true);
    
    std::cout << "Initial tensors:" << std::endl;
    std::cout << "Tensor a data: ";
    for (float val : a.data) std::cout << val << " ";
    std::cout << std::endl;
    
    std::cout << "Tensor b data: ";
    for (float val : b.data) std::cout << val << " ";
    std::cout << std::endl;
    
    // Perform operations
    Tensor c = a.add(b);
    Tensor d = c.relu();
    Tensor e = d.sum();
    
    std::cout << "\nAfter operations:" << std::endl;
    std::cout << "Sum result: " << e.data[0] << std::endl;
    
    // Backward pass
    e.backward();
    
    std::cout << "\nAfter backward pass:" << std::endl;
    std::cout << "Tensor a gradients: ";
    for (float grad : a.grad) std::cout << grad << " ";
    std::cout << std::endl;
    
    std::cout << "Tensor b gradients: ";
    for (float grad : b.grad) std::cout << grad << " ";
    std::cout << std::endl;
    
    return 0;
} 
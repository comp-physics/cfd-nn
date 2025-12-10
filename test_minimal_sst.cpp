#include "turbulence_transport.hpp"
#include "mesh.hpp"
#include "fields.hpp"
#include <iostream>

int main() {
    std::cout << "Creating SST model... ";
    nncfd::SSTKOmegaTransport model;
    std::cout << "OK\n";
    
    std::cout << "Setting parameters... ";
    model.set_nu(0.001);
    model.set_delta(1.0);
    std::cout << "OK\n";
    
    std::cout << "Model created successfully!\n";
    return 0;
}

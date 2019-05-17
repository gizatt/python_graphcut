#include <iostream>

int main(int argc, char** argv) {

    std::cout << "Calling 'python graphcut.py'..." << std::endl;
    return system("python graphcut.py -t");
}

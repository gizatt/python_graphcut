#include <iostream>

int main(int argc, char** argv) {

    std::cout << "Calling 'python graphcut.py'..." << std::endl;
    system("python graphcut.py")
    std::cout << "Done 'python graphcut.py'." << std::endl;

    return EXIT_SUCCESS;
}

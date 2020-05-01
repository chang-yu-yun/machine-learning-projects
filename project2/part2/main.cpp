#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <iomanip>

double nchoosek(int n, int k) {
    // base cases
    if (k == 0) {
        return 1.0;
    } else if (k == 1) {
        return n;
    }
    if (n - k < k) {
        return nchoosek(n, n - k);
    }
    double num = 1.0, den = 1.0;
    for (int i = 0; i < k; ++i) {
        num *= n - i;
        den *= k - i;
    }
    return num / den;
}

double calBinLikelihood(int total, int num1) {
    double mu = static_cast<double>(num1) / static_cast<double>(total);
    return nchoosek(total, num1) * pow(mu, num1) * pow(1.0 - mu, total - num1);
}

int main() {
    std::string inputStr;
    std::ifstream input;
    std::cout << "Please enter filename of input file: ";
    std::getline(std::cin, inputStr);
    input.open(inputStr.c_str());
    if (!input.is_open()) {
        std::cerr << "[Error] Cannot open the input file!" << std::endl;
        return -1;
    }
    int a, b;
    std::cout << "Please enter parameter a for initial beta prior: ";
    std::getline(std::cin, inputStr);
    a = std::stoi(inputStr);
    std::cout << "Please enter parameter b for initial beta prior: ";
    std::getline(std::cin, inputStr);
    b = std::stoi(inputStr);
    std::cout << std::endl;
    int curr = 0;
    while (std::getline(input, inputStr)) {
        ++curr;
        std::cout << "case " << curr << ": " << inputStr << std::endl;
        int total = inputStr.length();
        int num1 = 0;
        for (char c : inputStr) {
            if (c == '1') {
                ++num1;
            }
        }
        std::cout << "Likelihood: " << std::fixed << std::setprecision(16) << calBinLikelihood(total, num1) << std::endl;
        std::cout << "Beta prior:     a = " << a << "  b = " << b << std::endl;
        a += num1, b += total - num1;
        std::cout << "Beta posterior: a = " << a << "  b = " << b << std::endl;
        std::cout << std::endl;
    }
    input.close();
    return 0;
}


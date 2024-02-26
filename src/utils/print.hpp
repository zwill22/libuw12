//
// Created by Zack Williams on 29/01/2021.
//

#ifndef UW12_PRINT_HPP
#define UW12_PRINT_HPP

#include <iomanip>
#include <iostream>
#include <string>

namespace uw12::print {
    inline void print_character_line(const char character, const size_t n) {
        for (size_t i = 0; i < n; ++i) {
            std::cout << character;
        }
        std::cout << std::endl;
    }

    inline void print_header(const std::string &string, const size_t width = 48) {
        std::cout << std::endl;
        print_character_line('=', width);
        std::cout << " " << string << std::endl;
        print_character_line('=', width);
    }

    inline void print_result(const std::string &string, const double result, const size_t width = 32) {
        const auto size = string.size();
        if (size > width) {
            throw std::logic_error("Output string exceeds size of print block");
        }

        const auto pad = width - size;
        std::cout << " " << string << ":";
        for (size_t i = 0; i < pad; ++i) {
            std::cout << " ";
        }
        std::cout << std::setw(12) << std::fixed << std::right << std::setprecision(6)
                << result << std::endl;
    }
} // namespace uw12::print

#endif  // UW12_PRINT_HPP

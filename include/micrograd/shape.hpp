// Shape: ordered dims + numel. Row-major.
#pragma once
#include <cstdint>
#include <vector>
#include <numeric>
#include <stdexcept>

namespace micrograd {

struct Shape {
    std::vector<int64_t> dims;

    Shape() = default;
    explicit Shape(std::vector<int64_t> d) : dims(std::move(d)) {}
    Shape(std::initializer_list<int64_t> il) : dims(il) {}

    int64_t rank() const { return static_cast<int64_t>(dims.size()); }
    bool empty() const { return dims.empty(); }

    int64_t numel() const {
        if (dims.empty()) return 1;
        int64_t n = 1;
        for (int64_t d : dims) {
            if (d < 0) throw std::runtime_error("Shape has negative dim");
            n *= d;
        }
        return n;
    }

    bool operator==(const Shape& o) const { return dims == o.dims; }
    bool operator!=(const Shape& o) const { return dims != o.dims; }

    std::string to_string() const {
        std::string s = "(";
        for (size_t i = 0; i < dims.size(); ++i) {
            if (i) s += ", ";
            s += std::to_string(dims[i]);
        }
        s += ")";
        return s;
    }
};

}  // namespace micrograd

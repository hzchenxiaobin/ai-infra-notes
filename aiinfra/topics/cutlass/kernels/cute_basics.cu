// cute_basics.cu —— CuTe Layout/Tensor 基础练习
// 编译: nvcc -o cute_basics cute_basics.cu \
//   -I${CUTLASS_ROOT}/include -arch=sm_90a -std=c++17
// 运行: ./cute_basics

#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

int main() {
    std::cout << "=== CuTe Layout Basics ===" << std::endl;

    auto layout_col = make_layout(make_shape(4, 4), make_stride(1, 4));
    std::cout << "Column-Major Layout: " << layout_col << std::endl;

    auto layout_row = make_layout(make_shape(4, 4), make_stride(4, 1));
    std::cout << "Row-Major Layout:    " << layout_row << std::endl;

    std::cout << "\n=== Offset Comparison ===" << std::endl;
    std::cout << "Coord (2,3):" << std::endl;
    std::cout << "  ColMajor: " << layout_col(2, 3) << std::endl;
    std::cout << "  RowMajor: " << layout_row(2, 3) << std::endl;

    std::cout << "\n=== Tensor Access ===" << std::endl;
    float data[16] = {0};
    for (int i = 0; i < 16; ++i) data[i] = (float)i;

    auto tensor_col = make_tensor(data, layout_col);
    auto tensor_row = make_tensor(data, layout_row);

    std::cout << "data[14] = " << data[14] << std::endl;
    std::cout << "tensor_col(2,3) = " << tensor_col(2, 3) << std::endl;
    std::cout << "tensor_row(2,3) = " << tensor_row(2, 3) << std::endl;

    std::cout << "\n=== Tensor Slice ===" << std::endl;
    auto row2 = tensor_row(2, _);
    std::cout << "tensor_row(2, _) = [";
    for (int j = 0; j < 4; ++j) {
        std::cout << row2(j) << (j < 3 ? ", " : "");
    }
    std::cout << "]" << std::endl;

    std::cout << "\n=== Tensor Tile ===" << std::endl;
    auto tile = tensor_row(make_range(0, 2), make_range(0, 2));
    std::cout << "tensor_row(0:2, 0:2):" << std::endl;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            std::cout << tile(i, j) << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "\n=== Static Shape ===" << std::endl;
    auto static_layout = make_layout(make_shape(_8{}, _8{}), make_stride(_8{}, _1{}));
    std::cout << "Static Layout: " << static_layout << std::endl;
    std::cout << "Static(3,5) = " << static_layout(3, 5) << std::endl;

    return 0;
}

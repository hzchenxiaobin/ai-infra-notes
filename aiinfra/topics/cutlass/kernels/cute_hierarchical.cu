// cute_hierarchical.cu —— CuTe 嵌套 Layout 练习
// 编译: nvcc -o cute_hierarchical cute_hierarchical.cu \
//   -I${CUTLASS_ROOT}/include -arch=sm_90a -std=c++17

#include <cute/tensor.hpp>
#include <iostream>

using namespace cute;

int main() {
    std::cout << "=== Hierarchical Layout: Tiled Matrix ===" << std::endl;

    auto tiled_layout = make_layout(
        make_shape(make_shape(_4{}, _2{}), make_shape(_4{}, _2{})),
        make_stride(make_stride(_1{}, _4{}), make_stride(_8{}, _32{}))
    );
    std::cout << "Tiled Layout: " << tiled_layout << std::endl;

    auto offset = tiled_layout(make_coord(2, 1), make_coord(3, 1));
    std::cout << "Tile(1,1) elem(2,3) -> offset " << offset << std::endl;

    float data[64] = {0};
    for (int i = 0; i < 64; ++i) data[i] = (float)i;
    auto tensor = make_tensor(data, tiled_layout);

    auto tile_00 = tensor(make_coord(_, _0{}), make_coord(_, _0{}));
    std::cout << "\nTile(0,0):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << tile_00(i, j) << "\t";
        }
        std::cout << std::endl;
    }

    auto tile_11 = tensor(make_coord(_, _1{}), make_coord(_, _1{}));
    std::cout << "\nTile(1,1):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << tile_11(i, j) << "\t";
        }
        std::cout << std::endl;
    }

    return 0;
}

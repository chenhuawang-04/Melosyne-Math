#include <iostream>
#include <fast_math/bit_ops.h>

using namespace Melosyne;

int main() {
    BitSet<256> small{};
    BitSet<4096> medium{};
    BitSet<65536> large{};
    
    std::cout << "BitSet<256>:   " << small.kWords << " words\n";
    std::cout << "BitSet<4096>:  " << medium.kWords << " words\n";
    std::cout << "BitSet<65536>: " << large.kWords << " words\n";
    
    std::cout << "\nAVX2派发阈值: > 8 words\n";
    std::cout << "小位集会走AVX2: " << (small.kWords > 8 ? "是" : "否") << "\n";
    std::cout << "中等位集会走AVX2: " << (medium.kWords > 8 ? "是" : "否") << "\n";
    std::cout << "大位集会走AVX2: " << (large.kWords > 8 ? "是" : "否") << "\n";
    
    return 0;
}

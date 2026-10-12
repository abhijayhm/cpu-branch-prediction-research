#ifndef ANBA_FEATURES_EXT_H
#define ANBA_FEATURES_EXT_H

#include <cstdint>

namespace anba_ext
{

constexpr uint32_t PC_BITS = 16;
constexpr uint32_t LOCAL_BITS = 16;
constexpr uint32_t GLOBAL_BITS = 32;
constexpr uint32_t N_FEATURES = PC_BITS + LOCAL_BITS + GLOBAL_BITS;

inline void pack_features(uint64_t ip, uint64_t ghist64, uint16_t local_hist, int8_t* x)
{
  const uint64_t pc = ip >> 2;
  for (uint32_t i = 0; i < PC_BITS; i++) {
    x[i] = ((pc >> i) & 1ull) ? int8_t{1} : int8_t{-1};
  }
  for (uint32_t i = 0; i < LOCAL_BITS; i++) {
    x[PC_BITS + i] = ((local_hist >> i) & 1u) ? int8_t{1} : int8_t{-1};
  }
  for (uint32_t i = 0; i < GLOBAL_BITS; i++) {
    x[PC_BITS + LOCAL_BITS + i] = ((ghist64 >> i) & 1ull) ? int8_t{1} : int8_t{-1};
  }
}

} // namespace anba_ext

#endif

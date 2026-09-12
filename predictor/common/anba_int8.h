#ifndef ANBA_INT8_H
#define ANBA_INT8_H

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// Native INT8 MLP used by ChampSim predictors. No PyTorch at inference.
// Feature layout (32 x ±1):
//   [0:16)  PC bits of (ip >> 2)
//   [16:32) global history bits (LSB = most recent outcome)
namespace anba
{

constexpr uint32_t MAGIC = 0x414e4241; // 'ANBA'
constexpr uint32_t VERSION = 1;
constexpr uint32_t N_FEATURES = 32;
constexpr uint32_t HIST_BITS = 16;
constexpr uint32_t PC_BITS = 16;

enum class Arch : uint32_t { Dense16 = 0, Dense32_8 = 1, Perceptron = 2 };

inline void pack_features(uint64_t ip, uint32_t ghist, int8_t* x)
{
  const uint64_t pc = ip >> 2;
  for (uint32_t i = 0; i < PC_BITS; i++) {
    x[i] = ((pc >> i) & 1ull) ? int8_t{1} : int8_t{-1};
  }
  for (uint32_t i = 0; i < HIST_BITS; i++) {
    x[PC_BITS + i] = ((ghist >> i) & 1u) ? int8_t{1} : int8_t{-1};
  }
}

struct Layer {
  uint32_t in = 0;
  uint32_t out = 0;
  float weight_scale = 1.f;
  std::vector<int8_t> w;   // row-major [out, in]
  std::vector<int32_t> b;  // quantized bias in accumulator units
};

struct Model {
  Arch arch = Arch::Dense16;
  std::vector<Layer> layers;
  bool loaded = false;

  bool load(const char* path)
  {
    if (path == nullptr || path[0] == '\0') {
      return false;
    }
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
      return false;
    }
    uint32_t magic = 0, ver = 0, arch_u = 0, nfeat = 0, nlayer = 0;
    if (std::fread(&magic, 4, 1, f) != 1 || magic != MAGIC) {
      std::fclose(f);
      return false;
    }
    if (std::fread(&ver, 4, 1, f) != 1 || ver != VERSION) {
      std::fclose(f);
      return false;
    }
    if (std::fread(&arch_u, 4, 1, f) != 1) {
      std::fclose(f);
      return false;
    }
    if (std::fread(&nfeat, 4, 1, f) != 1 || nfeat != N_FEATURES) {
      std::fclose(f);
      return false;
    }
    if (std::fread(&nlayer, 4, 1, f) != 1 || nlayer == 0 || nlayer > 8) {
      std::fclose(f);
      return false;
    }
    arch = static_cast<Arch>(arch_u);
    layers.clear();
    layers.resize(nlayer);
    for (auto& layer : layers) {
      if (std::fread(&layer.in, 4, 1, f) != 1 || std::fread(&layer.out, 4, 1, f) != 1) {
        std::fclose(f);
        return false;
      }
      if (std::fread(&layer.weight_scale, 4, 1, f) != 1) {
        std::fclose(f);
        return false;
      }
      if (layer.in == 0 || layer.out == 0 || layer.in > 256 || layer.out > 256) {
        std::fclose(f);
        return false;
      }
      layer.w.resize(static_cast<size_t>(layer.out) * layer.in);
      layer.b.resize(layer.out);
      if (std::fread(layer.w.data(), 1, layer.w.size(), f) != layer.w.size()) {
        std::fclose(f);
        return false;
      }
      if (std::fread(layer.b.data(), 4, layer.b.size(), f) != layer.b.size()) {
        std::fclose(f);
        return false;
      }
    }
    std::fclose(f);
    loaded = true;
    return true;
  }

  // Hidden activations: sign of int32 accumulator (ternary ±1, 0 -> -1).
  bool predict(const int8_t* x) const
  {
    if (!loaded || layers.empty()) {
      return false;
    }
    std::vector<int8_t> cur(N_FEATURES);
    std::memcpy(cur.data(), x, N_FEATURES);
    std::vector<int32_t> acc;
    for (size_t li = 0; li < layers.size(); li++) {
      const auto& layer = layers[li];
      acc.assign(layer.out, 0);
      for (uint32_t o = 0; o < layer.out; o++) {
        int32_t s = layer.b[o];
        const int8_t* row = &layer.w[static_cast<size_t>(o) * layer.in];
        for (uint32_t i = 0; i < layer.in; i++) {
          s += static_cast<int32_t>(row[i]) * static_cast<int32_t>(cur[i]);
        }
        acc[o] = s;
      }
      if (li + 1 == layers.size()) {
        return acc[0] >= 0;
      }
      cur.resize(layer.out);
      for (uint32_t o = 0; o < layer.out; o++) {
        cur[o] = acc[o] >= 0 ? int8_t{1} : int8_t{-1};
      }
    }
    return false;
  }
};

} // namespace anba

#endif

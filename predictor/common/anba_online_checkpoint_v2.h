#ifndef ANBA_ONLINE_CHECKPOINT_V2_H
#define ANBA_ONLINE_CHECKPOINT_V2_H

#include <bitset>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace anba_ckpt2
{

constexpr uint32_t MAGIC = 0x414f4e32; // AON2
constexpr uint32_t VERSION = 2;
constexpr uint32_t N_FEAT = 64;
constexpr uint32_t GS_SIZE = 16384;
constexpr uint32_t LOCAL_SIZE = 4096;
constexpr uint32_t CHOOSER_SIZE = 16384;
constexpr uint32_t GSHARE_HIST_LEN = 14;

struct State {
  uint8_t gs_table[GS_SIZE]{};
  uint16_t gshare_hist = 0;
  uint64_t ghist64 = 0;
  uint16_t local_hist[LOCAL_SIZE]{};
  int32_t w[N_FEAT]{};
  int32_t b = 0;
  int8_t chooser[CHOOSER_SIZE]{};
  int8_t hysteresis[CHOOSER_SIZE]{};
  uint64_t train_updates = 0;
};

inline bool load(State& s, const char* path)
{
  if (!path || !path[0]) {
    return false;
  }
  std::FILE* f = std::fopen(path, "rb");
  if (!f) {
    return false;
  }
  uint32_t magic = 0, ver = 0;
  if (std::fread(&magic, 4, 1, f) != 1 || magic != MAGIC) {
    std::fclose(f);
    return false;
  }
  if (std::fread(&ver, 4, 1, f) != 1 || ver != VERSION) {
    std::fclose(f);
    return false;
  }
  auto rd = [&](void* p, size_t n) {
    return std::fread(p, 1, n, f) == n;
  };
  if (!rd(s.gs_table, GS_SIZE) || !rd(&s.gshare_hist, sizeof(uint16_t)) || !rd(&s.ghist64, sizeof(uint64_t)) ||
      !rd(s.local_hist, LOCAL_SIZE * sizeof(uint16_t)) || !rd(s.w, N_FEAT * sizeof(int32_t)) ||
      !rd(&s.b, sizeof(int32_t)) || !rd(s.chooser, CHOOSER_SIZE) || !rd(s.hysteresis, CHOOSER_SIZE) ||
      !rd(&s.train_updates, sizeof(uint64_t))) {
    std::fclose(f);
    return false;
  }
  std::fclose(f);
  return true;
}

inline bool save(const State& s, const char* path)
{
  if (!path || !path[0]) {
    return false;
  }
  std::FILE* f = std::fopen(path, "wb");
  if (!f) {
    return false;
  }
  const uint32_t magic = MAGIC;
  const uint32_t ver = VERSION;
  if (std::fwrite(&magic, 4, 1, f) != 1 || std::fwrite(&ver, 4, 1, f) != 1) {
    std::fclose(f);
    return false;
  }
  auto wr = [&](const void* p, size_t n) { return std::fwrite(p, 1, n, f) == n; };
  if (!wr(s.gs_table, GS_SIZE) || !wr(&s.gshare_hist, sizeof(uint16_t)) || !wr(&s.ghist64, sizeof(uint64_t)) ||
      !wr(s.local_hist, LOCAL_SIZE * sizeof(uint16_t)) || !wr(s.w, N_FEAT * sizeof(int32_t)) ||
      !wr(&s.b, sizeof(int32_t)) || !wr(s.chooser, CHOOSER_SIZE) || !wr(s.hysteresis, CHOOSER_SIZE) ||
      !wr(&s.train_updates, sizeof(uint64_t))) {
    std::fclose(f);
    return false;
  }
  std::fclose(f);
  return true;
}

} // namespace anba_ckpt2

#endif

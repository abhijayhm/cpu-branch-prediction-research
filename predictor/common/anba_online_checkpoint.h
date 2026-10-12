#ifndef ANBA_ONLINE_CHECKPOINT_H
#define ANBA_ONLINE_CHECKPOINT_H

#include <cstdint>
#include <cstdio>
#include <cstring>

namespace anba_online_ckpt
{

constexpr uint32_t MAGIC = 0x414f4e4c; // 'AONL'
constexpr uint32_t VERSION = 1;
constexpr uint32_t N_FEAT = 32;
constexpr uint32_t CHOOSER_SIZE = 16384;

struct State {
  int32_t w[N_FEAT]{};
  int32_t b = 0;
  int8_t chooser[CHOOSER_SIZE]{}; // + => trust online model at this PC bucket
  uint64_t train_updates = 0;
};

inline bool load(State& s, const char* path)
{
  if (path == nullptr || path[0] == '\0') {
    return false;
  }
  std::FILE* f = std::fopen(path, "rb");
  if (f == nullptr) {
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
  if (std::fread(s.w, sizeof(int32_t), N_FEAT, f) != N_FEAT) {
    std::fclose(f);
    return false;
  }
  if (std::fread(&s.b, sizeof(int32_t), 1, f) != 1) {
    std::fclose(f);
    return false;
  }
  if (std::fread(s.chooser, 1, CHOOSER_SIZE, f) != CHOOSER_SIZE) {
    std::fclose(f);
    return false;
  }
  if (std::fread(&s.train_updates, sizeof(uint64_t), 1, f) != 1) {
    std::fclose(f);
    return false;
  }
  std::fclose(f);
  return true;
}

inline bool save(const State& s, const char* path)
{
  if (path == nullptr || path[0] == '\0') {
    return false;
  }
  std::FILE* f = std::fopen(path, "wb");
  if (f == nullptr) {
    return false;
  }
  const uint32_t magic = MAGIC;
  const uint32_t ver = VERSION;
  if (std::fwrite(&magic, 4, 1, f) != 1 || std::fwrite(&ver, 4, 1, f) != 1) {
    std::fclose(f);
    return false;
  }
  if (std::fwrite(s.w, sizeof(int32_t), N_FEAT, f) != N_FEAT) {
    std::fclose(f);
    return false;
  }
  if (std::fwrite(&s.b, sizeof(int32_t), 1, f) != 1) {
    std::fclose(f);
    return false;
  }
  if (std::fwrite(s.chooser, 1, CHOOSER_SIZE, f) != CHOOSER_SIZE) {
    std::fclose(f);
    return false;
  }
  if (std::fwrite(&s.train_updates, sizeof(uint64_t), 1, f) != 1) {
    std::fclose(f);
    return false;
  }
  std::fclose(f);
  return true;
}

} // namespace anba_online_ckpt

#endif

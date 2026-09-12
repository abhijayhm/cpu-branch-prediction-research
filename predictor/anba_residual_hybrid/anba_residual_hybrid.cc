#include "anba_residual_hybrid.h"

#include <cstdlib>

void anba_residual_hybrid::initialize_branch_predictor()
{
  const char* path = std::getenv("ANBA_MODEL_PATH");
  nn_ready = model.load(path);
  nn_margin = DEFAULT_NN_MARGIN;
  if (const char* m = std::getenv("ANBA_NN_MARGIN")) {
    nn_margin = static_cast<int32_t>(std::strtol(m, nullptr, 10));
    if (nn_margin < 0) {
      nn_margin = DEFAULT_NN_MARGIN;
    }
  }
}

bool anba_residual_hybrid::predict_branch(champsim::address ip)
{
  const auto ctr = table[hash(ip)];
  const auto v = ctr.value();
  const bool conventional = v > (ctr.maximum / 2);
  const bool confident = (v == 0) || (v == ctr.maximum); // 00 or 11 for 2-bit
  if (confident || !nn_ready) {
    return conventional;
  }
  int8_t x[anba::N_FEATURES];
  anba::pack_features(ip.to<uint64_t>(), ghist, x);
  const int32_t acc = model.final_accumulator(x);
  if (std::abs(acc) < nn_margin) {
    return conventional;
  }
  return acc >= 0;
}

void anba_residual_hybrid::last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  (void)branch_target;
  (void)branch_type;
  table[hash(ip)] += taken ? 1 : -1;
  ghist = ((ghist << 1) | (taken ? 1u : 0u)) & ((1u << anba::HIST_BITS) - 1u);
}

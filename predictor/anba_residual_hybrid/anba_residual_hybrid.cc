#include "anba_residual_hybrid.h"

#include <cmath>
#include <cstdlib>

#include "anba_gate_stats.h"

void anba_residual_hybrid::initialize_branch_predictor()
{
  anba_gate::ensure_registered();
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
  last_uncertain = false;
  last_override = false;
  if (confident || !nn_ready) {
    return conventional;
  }
  last_uncertain = true;
  int8_t x[anba::N_FEATURES];
  anba::pack_features(ip.to<uint64_t>(), ghist, x);
  const int32_t acc = model.final_accumulator(x);
  last_nn_pred = acc >= 0;
  if (std::abs(acc) < nn_margin) {
    return conventional;
  }
  last_override = true;
  return last_nn_pred;
}

void anba_residual_hybrid::last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  (void)branch_target;
  (void)branch_type;
  if (last_uncertain) {
    anba_gate::counters().uncertain++;
    if (last_override) {
      anba_gate::counters().override_n++;
      if (last_nn_pred == taken) {
        anba_gate::counters().override_correct++;
      }
    }
  }
  table[hash(ip)] += taken ? 1 : -1;
  ghist = ((ghist << 1) | (taken ? 1u : 0u)) & ((1u << anba::HIST_BITS) - 1u);
}

anba_residual_hybrid::~anba_residual_hybrid() = default;

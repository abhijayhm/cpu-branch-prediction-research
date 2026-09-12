#include "nn_frozen.h"

void nn_frozen::initialize_branch_predictor()
{
  const char* path = std::getenv("ANBA_MODEL_PATH");
  ready = model.load(path);
}

bool nn_frozen::predict_branch(champsim::address ip)
{
  if (!ready) {
    return true; // fail-closed to taken if weights missing; not a silent random draw
  }
  int8_t x[anba::N_FEATURES];
  anba::pack_features(ip.to<uint64_t>(), ghist, x);
  return model.predict(x);
}

void nn_frozen::last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  (void)ip;
  (void)branch_target;
  (void)branch_type;
  ghist = ((ghist << 1) | (taken ? 1u : 0u)) & ((1u << anba::HIST_BITS) - 1u);
}

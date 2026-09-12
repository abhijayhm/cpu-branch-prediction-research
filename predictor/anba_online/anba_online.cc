#include "anba_online.h"

void anba_online::initialize_branch_predictor()
{
  const char* path = std::getenv("ANBA_MODEL_PATH");
  nn_ready = model.load(path);
  const char* cold = std::getenv("ANBA_COLDSTART_N");
  if (cold != nullptr && cold[0] != '\0') {
    coldstart_n = static_cast<uint64_t>(std::strtoull(cold, nullptr, 10));
  }
}

bool anba_online::predict_branch(champsim::address ip)
{
  const auto ctr = table[hash(ip)];
  const auto v = ctr.value();
  const bool conventional = v > (ctr.maximum / 2);
  if (seen < coldstart_n || !nn_ready) {
    last_nn = false;
    return conventional;
  }
  const bool confident = (v == 0) || (v == ctr.maximum);
  if (confident) {
    last_nn = false;
    return conventional;
  }
  anba::pack_features(ip.to<uint64_t>(), ghist, last_x);
  int32_t acc = residual_b;
  for (uint32_t i = 0; i < anba::N_FEATURES; i++) {
    acc += static_cast<int32_t>(residual_w[i]) * static_cast<int32_t>(last_x[i]);
  }
  const bool nn = model.predict(last_x);
  last_nn = true;
  // Residual vote: if the online perceptron is confident, XOR-shift the frozen bit.
  if (acc > 8) {
    return true;
  }
  if (acc < -8) {
    return false;
  }
  return nn;
}

void anba_online::last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  (void)branch_target;
  (void)branch_type;
  table[hash(ip)] += taken ? 1 : -1;
  if (last_nn) {
    const int8_t y = taken ? int8_t{1} : int8_t{-1};
    int32_t acc = residual_b;
    for (uint32_t i = 0; i < anba::N_FEATURES; i++) {
      acc += static_cast<int32_t>(residual_w[i]) * static_cast<int32_t>(last_x[i]);
    }
    const int8_t pred = acc >= 0 ? int8_t{1} : int8_t{-1};
    if (pred != y || (acc > -8 && acc < 8)) {
      for (uint32_t i = 0; i < anba::N_FEATURES; i++) {
        int32_t next = static_cast<int32_t>(residual_w[i]) + static_cast<int32_t>(y) * static_cast<int32_t>(last_x[i]);
        if (next > 127) {
          next = 127;
        }
        if (next < -127) {
          next = -127;
        }
        residual_w[i] = static_cast<int8_t>(next);
      }
      residual_b += y;
    }
  }
  last_nn = false;
  ghist = ((ghist << 1) | (taken ? 1u : 0u)) & ((1u << anba::HIST_BITS) - 1u);
  seen++;
}

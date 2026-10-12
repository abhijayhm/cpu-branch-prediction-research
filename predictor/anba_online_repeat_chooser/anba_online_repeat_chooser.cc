#include "anba_online_repeat_chooser.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "anba_gate_stats.h"
#include "anba_int8.h"
#include "anba_online_checkpoint.h"
#include "anba_online_epoch_stats.h"

void anba_online_repeat_chooser::train_on_mispredict(const int8_t* x, bool taken)
{
  const int8_t y = taken ? int8_t{1} : int8_t{-1};
  for (uint32_t i = 0; i < anba::N_FEATURES; i++) {
    int32_t next = ol_w[i] + static_cast<int32_t>(y) * static_cast<int32_t>(x[i]);
    if (next > 32767) {
      next = 32767;
    }
    if (next < -32767) {
      next = -32767;
    }
    ol_w[i] = next;
  }
  ol_b += y;
  anba_epoch::counters().train_updates++;
}

void anba_online_repeat_chooser::update_chooser(std::size_t h, bool bim_ok, bool model_ok)
{
  int32_t t = chooser[h];
  if (model_ok && !bim_ok) {
    t += 3;
  } else if (bim_ok && !model_ok) {
    t -= 3;
  } else if (model_ok && bim_ok) {
    t += 1;
  } else {
    t -= 1;
  }
  if (t > 127) {
    t = 127;
  }
  if (t < -127) {
    t = -127;
  }
  chooser[h] = static_cast<int8_t>(t);
}

void anba_online_repeat_chooser::initialize_branch_predictor()
{
  anba_gate::ensure_registered();
  anba_epoch::ensure_registered();
  anba_epoch::read_epoch_from_env();
  std::memset(ol_w, 0, sizeof(ol_w));
  ol_b = 0;
  chooser.fill(0);
  table.fill({});
  ghist = 0;
  if (const char* ck = std::getenv("ANBA_CHECKPOINT_PATH")) {
    anba_online_ckpt::State st{};
    if (anba_online_ckpt::load(st, ck)) {
      std::memcpy(ol_w, st.w, sizeof(ol_w));
      ol_b = st.b;
      std::memcpy(chooser.data(), st.chooser, chooser.size());
      anba_epoch::counters().train_updates = st.train_updates;
    }
  }
}

bool anba_online_repeat_chooser::predict_branch(champsim::address ip)
{
  last_h = hash(ip);
  const auto ctr = table[last_h];
  const auto v = ctr.value();
  last_bim_pred = v > (ctr.maximum / 2);
  const bool confident = (v == 0) || (v == ctr.maximum);
  last_uncertain = false;
  last_override = false;
  anba::pack_features(ip.to<uint64_t>(), ghist, last_x);
  const int32_t acc = ol_b;
  int32_t macc = acc;
  for (uint32_t i = 0; i < anba::N_FEATURES; i++) {
    macc += ol_w[i] * static_cast<int32_t>(last_x[i]);
  }
  last_model_pred = macc >= 0;
  if (confident) {
    last_pred = last_bim_pred;
    return last_pred;
  }
  last_uncertain = true;
  const int32_t trust = chooser[last_h];
  if (trust >= CHOOSER_TRUST && std::abs(macc) >= MODEL_MARGIN) {
    last_override = true;
    last_pred = last_model_pred;
    return last_pred;
  }
  last_pred = last_bim_pred;
  return last_pred;
}

void anba_online_repeat_chooser::last_branch_result(
    champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  (void)branch_target;
  (void)branch_type;
  if (last_uncertain) {
    anba_gate::counters().uncertain++;
    if (last_override) {
      anba_gate::counters().override_n++;
      if (last_pred == taken) {
        anba_gate::counters().override_correct++;
      }
    }
  }
  const bool bim_ok = last_bim_pred == taken;
  const bool model_ok = last_model_pred == taken;
  if (!bim_ok) {
    train_on_mispredict(last_x, taken);
  }
  update_chooser(last_h, bim_ok, model_ok);
  table[last_h] += taken ? 1 : -1;
  ghist = ((ghist << 1) | (taken ? 1u : 0u)) & ((1u << anba::HIST_BITS) - 1u);
}

anba_online_repeat_chooser::~anba_online_repeat_chooser()
{
  if (const char* ck = std::getenv("ANBA_CHECKPOINT_PATH")) {
    anba_online_ckpt::State st{};
    std::memcpy(st.w, ol_w, sizeof(ol_w));
    st.b = ol_b;
    std::memcpy(st.chooser, chooser.data(), chooser.size());
    st.train_updates = anba_epoch::counters().train_updates;
    anba_online_ckpt::save(st, ck);
  }
}

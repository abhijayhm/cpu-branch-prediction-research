#include "anba_online_repeat_vote.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "anba_gate_stats.h"
#include "anba_int8.h"
#include "anba_online_checkpoint.h"
#include "anba_online_epoch_stats.h"

void anba_online_repeat_vote::train_on_mispredict(const int8_t* x, bool taken)
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

void anba_online_repeat_vote::initialize_branch_predictor()
{
  anba_gate::ensure_registered();
  anba_epoch::ensure_registered();
  anba_epoch::read_epoch_from_env();
  std::memset(ol_w, 0, sizeof(ol_w));
  ol_b = 0;
  table.fill({});
  ghist = 0;
  if (const char* ck = std::getenv("ANBA_CHECKPOINT_PATH")) {
    anba_online_ckpt::State st{};
    if (anba_online_ckpt::load(st, ck)) {
      std::memcpy(ol_w, st.w, sizeof(ol_w));
      ol_b = st.b;
      anba_epoch::counters().train_updates = st.train_updates;
    }
  }
}

bool anba_online_repeat_vote::predict_branch(champsim::address ip)
{
  const auto h = hash(ip);
  const auto ctr = table[h];
  const auto v = ctr.value();
  last_bim_pred = v > (ctr.maximum / 2);
  const bool confident = (v == 0) || (v == ctr.maximum);
  last_uncertain = false;
  last_override = false;
  anba::pack_features(ip.to<uint64_t>(), ghist, last_x);
  int32_t macc = ol_b;
  for (uint32_t i = 0; i < anba::N_FEATURES; i++) {
    macc += ol_w[i] * static_cast<int32_t>(last_x[i]);
  }
  if (confident) {
    last_pred = last_bim_pred;
    return last_pred;
  }
  last_uncertain = true;
  // Bimodal weak confidence: 01 -> -1, 10 -> +1 toward taken side
  int32_t bim_vote = static_cast<int32_t>(v) - static_cast<int32_t>(ctr.maximum) / 2;
  const int32_t combined = bim_vote * 4 + macc / 2;
  const bool vote_taken = combined >= 0;
  // Use the combined vote direction when overriding (not raw model_pred alone).
  if (std::abs(macc) >= MODEL_MARGIN && std::abs(combined) >= VOTE_MARGIN && vote_taken != last_bim_pred) {
    last_override = true;
    last_pred = vote_taken;
    return last_pred;
  }
  last_pred = last_bim_pred;
  return last_pred;
}

void anba_online_repeat_vote::last_branch_result(
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
  if (last_bim_pred != taken) {
    train_on_mispredict(last_x, taken);
  }
  table[hash(ip)] += taken ? 1 : -1;
  ghist = ((ghist << 1) | (taken ? 1u : 0u)) & ((1u << anba::HIST_BITS) - 1u);
}

anba_online_repeat_vote::~anba_online_repeat_vote()
{
  if (const char* ck = std::getenv("ANBA_CHECKPOINT_PATH")) {
    anba_online_ckpt::State st{};
    std::memcpy(st.w, ol_w, sizeof(ol_w));
    st.b = ol_b;
    std::memset(st.chooser, 0, sizeof(st.chooser));
    st.train_updates = anba_epoch::counters().train_updates;
    anba_online_ckpt::save(st, ck);
  }
}

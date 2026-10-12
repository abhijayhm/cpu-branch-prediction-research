#include "anba_transformer_hybrid.h"

#include <cmath>
#include <cstdio>

#include "anba_gate_stats.h"

void anba_transformer_hybrid::initialize_branch_predictor()
{
  anba_gate::ensure_registered();
  const char* path = std::getenv("ANBA_MODEL_PATH");
  nn_ready = model.load(path);
  if (!nn_ready) {
    std::fprintf(stderr, "ANBA_TRANSFORMER_LOAD failed path=%s\n", path ? path : "(null)");
  }
  nn_margin = model.margin;
  if (const char* m = std::getenv("ANBA_NN_MARGIN")) {
    nn_margin = static_cast<float>(std::strtod(m, nullptr));
    if (nn_margin < 0.f) {
      nn_margin = model.margin;
    }
  }
  ring_len = 0;
}

bool anba_transformer_hybrid::predict_branch(champsim::address ip)
{
  const auto ctr = table[hash(ip)];
  const auto v = ctr.value();
  const bool conventional = v > (ctr.maximum / 2);
  const bool confident = (v == 0) || (v == ctr.maximum);
  last_uncertain = false;
  last_override = false;
  if (confident || !nn_ready) {
    return conventional;
  }
  last_uncertain = true;
  uint32_t tokens[anba_t::SEQ_LEN];
  const uint32_t pad = anba_t::VOCAB_CURRENT_OFFSET;
  const uint32_t need = model.seq_len;
  const uint32_t hist = std::min(ring_len, need - 1);
  const uint32_t pad_n = need - 1 - hist;
  uint32_t pos = 0;
  for (uint32_t i = 0; i < pad_n; i++) {
    tokens[pos++] = pad;
  }
  for (uint32_t i = 0; i < hist; i++) {
    tokens[pos++] = ring[ring_len - hist + i];
  }
  tokens[pos++] = anba_t::current_token(ip.to<uint64_t>());
  const float logit = model.forward_logit(tokens, static_cast<int>(need));
  last_nn_pred = logit >= 0.f;
  if (std::fabs(logit) < nn_margin) {
    return conventional;
  }
  last_override = true;
  return last_nn_pred;
}

void anba_transformer_hybrid::last_branch_result(
    champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  (void)branch_target;
  (void)branch_type;
  if (last_uncertain) {
    stat_uncertain++;
    anba_gate::counters().uncertain++;
    if (last_override) {
      stat_override++;
      anba_gate::counters().override_n++;
      if (last_nn_pred == taken) {
        stat_override_correct++;
        anba_gate::counters().override_correct++;
      }
    }
  }
  table[hash(ip)] += taken ? 1 : -1;
  const uint32_t tok = anba_t::branch_token(ip.to<uint64_t>(), taken);
  if (ring_len < anba_t::SEQ_LEN - 1) {
    ring[ring_len++] = tok;
  } else {
    for (uint32_t i = 1; i < ring_len; i++) {
      ring[i - 1] = ring[i];
    }
    ring[ring_len - 1] = tok;
  }
}

anba_transformer_hybrid::~anba_transformer_hybrid() = default;

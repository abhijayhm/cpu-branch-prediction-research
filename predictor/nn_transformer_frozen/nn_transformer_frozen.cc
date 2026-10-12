#include "nn_transformer_frozen.h"

void nn_transformer_frozen::initialize_branch_predictor()
{
  const char* path = std::getenv("ANBA_MODEL_PATH");
  ready = model.load(path);
  ring_len = 0;
}

bool nn_transformer_frozen::predict_branch(champsim::address ip)
{
  if (!ready) {
    return true;
  }
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
  return model.predict(tokens, static_cast<int>(need));
}

void nn_transformer_frozen::last_branch_result(
    champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  (void)branch_target;
  (void)branch_type;
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

#include "anba_gshare_online.h"

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "anba_features_ext.h"
#include "anba_gate_stats.h"
#include "anba_online_checkpoint_v2.h"
#include "anba_online_epoch_stats.h"

std::size_t anba_gshare_online::gs_table_hash(champsim::address ip, std::bitset<GLOBAL_HISTORY_LENGTH> bh_vector)
{
  constexpr champsim::data::bits LOG2_HISTORY_TABLE_SIZE{champsim::lg2(GS_HISTORY_TABLE_SIZE)};
  constexpr champsim::data::bits LENGTH{GLOBAL_HISTORY_LENGTH};
  std::size_t hash = bh_vector.to_ullong();
  hash ^= ip.slice<LOG2_HISTORY_TABLE_SIZE, champsim::data::bits{}>().to<std::size_t>();
  hash ^= ip.slice<LOG2_HISTORY_TABLE_SIZE + LENGTH, LENGTH>().to<std::size_t>();
  hash ^= ip.slice<LOG2_HISTORY_TABLE_SIZE + 2 * LENGTH, 2 * LENGTH>().to<std::size_t>();
  return hash % GS_HISTORY_TABLE_SIZE;
}

bool anba_gshare_online::gshare_predict(champsim::address ip) const
{
  const auto gs_hash = gs_table_hash(ip, branch_history_vector);
  const auto value = gs_history_table[gs_hash];
  return value.value() >= (value.maximum / 2);
}

void anba_gshare_online::train_on_gshare_miss(const int8_t* x, bool taken)
{
  const int8_t y = taken ? int8_t{1} : int8_t{-1};
  const int32_t rate = 1 << learn_shift;
  for (uint32_t i = 0; i < anba_ext::N_FEATURES; i++) {
    int32_t next = ol_w[i] + rate * static_cast<int32_t>(y) * static_cast<int32_t>(x[i]);
    if (next > 32767) {
      next = 32767;
    }
    if (next < -32767) {
      next = -32767;
    }
    ol_w[i] = next;
  }
  ol_b += rate * y;
  anba_epoch::counters().train_updates++;
}

void anba_gshare_online::update_chooser(std::size_t h, bool gshare_ok, bool model_ok)
{
  int32_t t = chooser[h];
  if (model_ok && !gshare_ok) {
    t += 3;
  } else if (gshare_ok && !model_ok) {
    t -= 3;
  } else if (model_ok && gshare_ok) {
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

void anba_gshare_online::initialize_branch_predictor()
{
  anba_gate::ensure_registered();
  anba_epoch::ensure_registered();
  anba_epoch::read_epoch_from_env();
  if (const char* v = std::getenv("ANBA_CHOOSER_HIGH")) {
    chooser_high = static_cast<int32_t>(std::strtol(v, nullptr, 10));
  }
  if (const char* v = std::getenv("ANBA_CHOOSER_LOW")) {
    chooser_low = static_cast<int32_t>(std::strtol(v, nullptr, 10));
  }
  if (const char* v = std::getenv("ANBA_MODEL_MARGIN")) {
    model_margin = static_cast<int32_t>(std::strtol(v, nullptr, 10));
  }
  if (const char* v = std::getenv("ANBA_LEARN_SHIFT")) {
    learn_shift = static_cast<int32_t>(std::strtol(v, nullptr, 10));
    if (learn_shift < 0) {
      learn_shift = 0;
    }
    if (learn_shift > 4) {
      learn_shift = 4;
    }
  }
  const bool persist = std::getenv("ANBA_PERSIST_STATE") && std::getenv("ANBA_PERSIST_STATE")[0] == '1';
  const char* ck = std::getenv("ANBA_CHECKPOINT_PATH");
  if (persist && ck) {
    anba_ckpt2::State st{};
    if (anba_ckpt2::load(st, ck)) {
      for (std::size_t i = 0; i < GS_HISTORY_TABLE_SIZE; i++) {
        gs_history_table[i] = champsim::msl::fwcounter<COUNTER_BITS>{};
        for (int k = 0; k < st.gs_table[i]; k++) {
          gs_history_table[i] += 1;
        }
      }
      branch_history_vector = std::bitset<GLOBAL_HISTORY_LENGTH>(st.gshare_hist);
      ghist64 = st.ghist64;
      for (std::size_t i = 0; i < LOCAL_TABLE_SIZE; i++) {
        local_hist_table[i] = st.local_hist[i];
      }
      std::memcpy(ol_w, st.w, sizeof(ol_w));
      ol_b = st.b;
      std::memcpy(chooser.data(), st.chooser, chooser.size());
      std::memcpy(hysteresis.data(), st.hysteresis, hysteresis.size());
      anba_epoch::counters().train_updates = st.train_updates;
      return;
    }
  }
  gs_history_table.fill({});
  branch_history_vector = {};
  ghist64 = 0;
  local_hist_table.fill(0);
  std::memset(ol_w, 0, sizeof(ol_w));
  ol_b = 0;
  chooser.fill(0);
  hysteresis.fill(0);
}

bool anba_gshare_online::predict_branch(champsim::address ip)
{
  last_h = pc_hash(ip);
  last_gshare_pred = gshare_predict(ip);
  last_override = false;
  last_model_active = false;
  const uint16_t lh = local_hist_table[local_hash(ip)];
  anba_ext::pack_features(ip.to<uint64_t>(), ghist64, lh, last_x);
  int32_t macc = ol_b;
  for (uint32_t i = 0; i < anba_ext::N_FEATURES; i++) {
    macc += ol_w[i] * static_cast<int32_t>(last_x[i]);
  }
  last_model_pred = macc >= 0;
  const int32_t trust = chooser[last_h];
  int8_t& hyst = hysteresis[last_h];
  if (trust >= chooser_high && std::abs(macc) >= model_margin) {
    hyst = 1;
  } else if (trust <= chooser_low) {
    hyst = -1;
  }
  last_model_active = std::abs(macc) >= model_margin;
  if (hyst > 0 && last_model_active) {
    last_override = true;
    last_pred = last_model_pred;
    return last_pred;
  }
  last_pred = last_gshare_pred;
  return last_pred;
}

void anba_gshare_online::last_branch_result(
    champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  (void)branch_target;
  (void)branch_type;
  if (last_model_active) {
    anba_gate::counters().uncertain++;
    if (last_override) {
      anba_gate::counters().override_n++;
      if (last_pred == taken) {
        anba_gate::counters().override_correct++;
      }
    }
  }
  const bool gshare_ok = last_gshare_pred == taken;
  const bool model_ok = last_model_pred == taken;
  if (!gshare_ok) {
    train_on_gshare_miss(last_x, taken);
  }
  update_chooser(last_h, gshare_ok, model_ok);

  const auto gs_hash = gs_table_hash(ip, branch_history_vector);
  gs_history_table[gs_hash] += taken ? 1 : -1;
  branch_history_vector <<= 1;
  branch_history_vector[0] = taken;
  ghist64 = ((ghist64 << 1) | (taken ? 1ull : 0ull)) & 0xffffffffffffffffull;
  const std::size_t lh_idx = local_hash(ip);
  local_hist_table[lh_idx] = static_cast<uint16_t>(((local_hist_table[lh_idx] << 1) | (taken ? 1u : 0u)) & 0xffffu);

  last_override = false;
}

anba_gshare_online::~anba_gshare_online()
{
  if (const char* ck = std::getenv("ANBA_CHECKPOINT_PATH")) {
    anba_ckpt2::State st{};
    for (std::size_t i = 0; i < GS_HISTORY_TABLE_SIZE; i++) {
      st.gs_table[i] = static_cast<uint8_t>(gs_history_table[i].value());
    }
    st.gshare_hist = static_cast<uint16_t>(branch_history_vector.to_ullong());
    st.ghist64 = ghist64;
    for (std::size_t i = 0; i < LOCAL_TABLE_SIZE; i++) {
      st.local_hist[i] = local_hist_table[i];
    }
    std::memcpy(st.w, ol_w, sizeof(ol_w));
    st.b = ol_b;
    std::memcpy(st.chooser, chooser.data(), chooser.size());
    std::memcpy(st.hysteresis, hysteresis.data(), hysteresis.size());
    st.train_updates = anba_epoch::counters().train_updates;
    anba_ckpt2::save(st, ck);
  }
}

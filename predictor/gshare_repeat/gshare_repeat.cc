#include "gshare_repeat.h"

#include <cstdlib>

#include "anba_online_checkpoint_v2.h"

std::size_t gshare_repeat::gs_table_hash(champsim::address ip, std::bitset<GLOBAL_HISTORY_LENGTH> bh_vector)
{
  constexpr champsim::data::bits LOG2_HISTORY_TABLE_SIZE{champsim::lg2(GS_HISTORY_TABLE_SIZE)};
  constexpr champsim::data::bits LENGTH{GLOBAL_HISTORY_LENGTH};
  std::size_t hash = bh_vector.to_ullong();
  hash ^= ip.slice<LOG2_HISTORY_TABLE_SIZE, champsim::data::bits{}>().to<std::size_t>();
  hash ^= ip.slice<LOG2_HISTORY_TABLE_SIZE + LENGTH, LENGTH>().to<std::size_t>();
  hash ^= ip.slice<LOG2_HISTORY_TABLE_SIZE + 2 * LENGTH, 2 * LENGTH>().to<std::size_t>();
  return hash % GS_HISTORY_TABLE_SIZE;
}

void gshare_repeat::initialize_branch_predictor()
{
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
      return;
    }
  }
  gs_history_table.fill({});
  branch_history_vector = {};
}

bool gshare_repeat::predict_branch(champsim::address ip)
{
  const auto gs_hash = gs_table_hash(ip, branch_history_vector);
  const auto value = gs_history_table[gs_hash];
  return value.value() >= (value.maximum / 2);
}

void gshare_repeat::last_branch_result(
    champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  (void)branch_target;
  (void)branch_type;
  const auto gs_hash = gs_table_hash(ip, branch_history_vector);
  gs_history_table[gs_hash] += taken ? 1 : -1;
  branch_history_vector <<= 1;
  branch_history_vector[0] = taken;
}

gshare_repeat::~gshare_repeat()
{
  if (const char* ck = std::getenv("ANBA_CHECKPOINT_PATH")) {
    anba_ckpt2::State st{};
    for (std::size_t i = 0; i < GS_HISTORY_TABLE_SIZE; i++) {
      st.gs_table[i] = static_cast<uint8_t>(gs_history_table[i].value());
    }
    st.gshare_hist = static_cast<uint16_t>(branch_history_vector.to_ullong());
    anba_ckpt2::save(st, ck);
  }
}

#ifndef BRANCH_ANBA_GSHARE_ONLINE_H
#define BRANCH_ANBA_GSHARE_ONLINE_H

#include <array>
#include <bitset>
#include <cstdint>

#include "address.h"
#include "modules.h"
#include "msl/fwcounter.h"

// gshare base + online perceptron on gshare misses; per-PC chooser w/ hysteresis.
class anba_gshare_online : champsim::modules::branch_predictor
{
  static constexpr std::size_t GLOBAL_HISTORY_LENGTH = 14;
  static constexpr std::size_t COUNTER_BITS = 2;
  static constexpr std::size_t GS_HISTORY_TABLE_SIZE = 16384;
  static constexpr std::size_t LOCAL_TABLE_SIZE = 4096;
  static constexpr std::size_t LOCAL_PRIME = 4093;
  static constexpr std::size_t CHOOSER_SIZE = 16384;
  static constexpr std::size_t CHOOSER_PRIME = 16381;

  std::bitset<GLOBAL_HISTORY_LENGTH> branch_history_vector;
  std::array<champsim::msl::fwcounter<COUNTER_BITS>, GS_HISTORY_TABLE_SIZE> gs_history_table{};
  std::array<uint16_t, LOCAL_TABLE_SIZE> local_hist_table{};
  uint64_t ghist64 = 0;

  int32_t ol_w[64]{};
  int32_t ol_b = 0;
  std::array<int8_t, CHOOSER_SIZE> chooser{};
  std::array<int8_t, CHOOSER_SIZE> hysteresis{};

  int32_t chooser_high = 8;
  int32_t chooser_low = 3;
  int32_t model_margin = 12;
  int32_t learn_shift = 0;

  bool last_override = false;
  bool last_model_active = false;
  bool last_gshare_pred = false;
  bool last_model_pred = false;
  bool last_pred = false;
  int8_t last_x[64]{};
  std::size_t last_h = 0;

  [[nodiscard]] static std::size_t gs_table_hash(champsim::address ip, std::bitset<GLOBAL_HISTORY_LENGTH> bh);
  [[nodiscard]] static std::size_t pc_hash(champsim::address ip) { return ip.to<unsigned long>() % CHOOSER_PRIME; }
  [[nodiscard]] static std::size_t local_hash(champsim::address ip) { return ip.to<unsigned long>() % LOCAL_PRIME; }

  [[nodiscard]] bool gshare_predict(champsim::address ip) const;
  void train_on_gshare_miss(const int8_t* x, bool taken);
  void update_chooser(std::size_t h, bool gshare_ok, bool model_ok);

public:
  using branch_predictor::branch_predictor;

  void initialize_branch_predictor();
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
  ~anba_gshare_online();
};

#endif

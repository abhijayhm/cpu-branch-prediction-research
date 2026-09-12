#ifndef BRANCH_ANBA_RESIDUAL_HYBRID_H
#define BRANCH_ANBA_RESIDUAL_HYBRID_H

#include <array>
#include <cstdint>
#include <cstdlib>

#include "address.h"
#include "anba_int8.h"
#include "modules.h"
#include "msl/fwcounter.h"

// Safer hybrid gate (residual specialist):
//   2-bit bimodal 00 / 11 -> conventional (bimodal), no NN query
//   2-bit bimodal 01 / 10 -> query NN; override bimodal only if NN margin high
//   else keep bimodal prediction
// Default margin: ANBA_NN_MARGIN=8 (INT8 accumulator units).
class anba_residual_hybrid : champsim::modules::branch_predictor
{
  static constexpr std::size_t TABLE_SIZE = 16384;
  static constexpr std::size_t PRIME = 16381;
  static constexpr std::size_t BITS = 2;
  static constexpr int32_t DEFAULT_NN_MARGIN = 8;

  std::array<champsim::msl::fwcounter<BITS>, TABLE_SIZE> table{};
  anba::Model model{};
  uint32_t ghist = 0;
  bool nn_ready = false;
  int32_t nn_margin = DEFAULT_NN_MARGIN;

  [[nodiscard]] static constexpr auto hash(champsim::address ip) { return ip.to<unsigned long>() % PRIME; }

public:
  using branch_predictor::branch_predictor;

  void initialize_branch_predictor();
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
};

#endif

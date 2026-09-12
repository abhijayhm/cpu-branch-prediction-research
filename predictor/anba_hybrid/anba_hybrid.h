#ifndef BRANCH_ANBA_HYBRID_H
#define BRANCH_ANBA_HYBRID_H

#include <array>
#include <cstdint>
#include <cstdlib>

#include "address.h"
#include "anba_int8.h"
#include "modules.h"
#include "msl/fwcounter.h"

// Confidence-gated hybrid:
//   2-bit bimodal state 00 / 11 -> conventional (bimodal)
//   2-bit bimodal state 01 / 10 -> neural
// Neural weights are frozen (offline INT8).
class anba_hybrid : champsim::modules::branch_predictor
{
  static constexpr std::size_t TABLE_SIZE = 16384;
  static constexpr std::size_t PRIME = 16381;
  static constexpr std::size_t BITS = 2;

  std::array<champsim::msl::fwcounter<BITS>, TABLE_SIZE> table{};
  anba::Model model{};
  uint32_t ghist = 0;
  bool nn_ready = false;

  [[nodiscard]] static constexpr auto hash(champsim::address ip) { return ip.to<unsigned long>() % PRIME; }

public:
  using branch_predictor::branch_predictor;

  void initialize_branch_predictor();
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
};

#endif

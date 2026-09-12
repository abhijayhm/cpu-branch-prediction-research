#ifndef BRANCH_ANBA_INSTRUMENTED_H
#define BRANCH_ANBA_INSTRUMENTED_H

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "address.h"
#include "modules.h"
#include "msl/fwcounter.h"

// Bimodal baseline that also logs PC, global history, and outcome.
// Used to verify instrumentation and to dump training features.
class instrumented : champsim::modules::branch_predictor
{
  static constexpr std::size_t TABLE_SIZE = 16384;
  static constexpr std::size_t PRIME = 16381;
  static constexpr std::size_t BITS = 2;
  static constexpr std::size_t HIST_BITS = 16;

  std::array<champsim::msl::fwcounter<BITS>, TABLE_SIZE> table{};
  uint32_t ghist = 0;
  std::FILE* log = nullptr;
  uint64_t records = 0;
  uint64_t cap = 0;

  [[nodiscard]] static constexpr auto hash(champsim::address ip) { return ip.to<unsigned long>() % PRIME; }

public:
  using branch_predictor::branch_predictor;

  void initialize_branch_predictor();
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
};

#endif

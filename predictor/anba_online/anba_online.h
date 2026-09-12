#ifndef BRANCH_ANBA_ONLINE_H
#define BRANCH_ANBA_ONLINE_H

#include <array>
#include <cstdint>
#include <cstdlib>

#include "address.h"
#include "anba_int8.h"
#include "modules.h"
#include "msl/fwcounter.h"

// Cold-start / online mode:
//   Until ANBA_COLDSTART_N branches have been seen, use conventional 2-bit only.
//   After that, use the same 00/11 conventional, 01/10 neural gate.
//   Neural path applies a tiny perceptron-style INT8 update (frozen layers stay
//   as exported; only a 32-wide output perceptron residual is trained online).
class anba_online : champsim::modules::branch_predictor
{
  static constexpr std::size_t TABLE_SIZE = 16384;
  static constexpr std::size_t PRIME = 16381;
  static constexpr std::size_t BITS = 2;

  std::array<champsim::msl::fwcounter<BITS>, TABLE_SIZE> table{};
  anba::Model model{};
  std::array<int8_t, anba::N_FEATURES> residual_w{};
  int32_t residual_b = 0;
  uint32_t ghist = 0;
  uint64_t seen = 0;
  uint64_t coldstart_n = 2048;
  bool nn_ready = false;
  int8_t last_x[anba::N_FEATURES]{};
  bool last_nn = false;

  [[nodiscard]] static constexpr auto hash(champsim::address ip) { return ip.to<unsigned long>() % PRIME; }

public:
  using branch_predictor::branch_predictor;

  void initialize_branch_predictor();
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
};

#endif

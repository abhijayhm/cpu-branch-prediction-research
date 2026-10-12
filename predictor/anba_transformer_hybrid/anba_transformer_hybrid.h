#ifndef BRANCH_ANBA_TRANSFORMER_HYBRID_H
#define BRANCH_ANBA_TRANSFORMER_HYBRID_H

#include <array>
#include <cstdint>
#include <cstdlib>

#include "address.h"
#include "anba_transformer.h"
#include "modules.h"
#include "msl/fwcounter.h"

class anba_transformer_hybrid : champsim::modules::branch_predictor
{
  static constexpr std::size_t TABLE_SIZE = 16384;
  static constexpr std::size_t PRIME = 16381;
  static constexpr std::size_t BITS = 2;

  std::array<champsim::msl::fwcounter<BITS>, TABLE_SIZE> table{};
  anba_t::Model model{};
  std::array<uint32_t, anba_t::SEQ_LEN> ring{};
  uint32_t ring_len = 0;
  bool nn_ready = false;
  float nn_margin = 1.f;

  uint64_t stat_uncertain = 0;
  uint64_t stat_override = 0;
  uint64_t stat_override_correct = 0;
  bool last_uncertain = false;
  bool last_override = false;
  bool last_nn_pred = false;

  [[nodiscard]] static constexpr auto hash(champsim::address ip) { return ip.to<unsigned long>() % PRIME; }

public:
  using branch_predictor::branch_predictor;

  void initialize_branch_predictor();
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
  ~anba_transformer_hybrid();
};

#endif

#ifndef BRANCH_NN_TRANSFORMER_FROZEN_H
#define BRANCH_NN_TRANSFORMER_FROZEN_H

#include <array>
#include <cstdint>
#include <cstdlib>

#include "address.h"
#include "anba_transformer.h"
#include "modules.h"

class nn_transformer_frozen : champsim::modules::branch_predictor
{
  anba_t::Model model{};
  std::array<uint32_t, anba_t::SEQ_LEN> ring{};
  uint32_t ring_len = 0;
  bool ready = false;

public:
  using branch_predictor::branch_predictor;

  void initialize_branch_predictor();
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
};

#endif

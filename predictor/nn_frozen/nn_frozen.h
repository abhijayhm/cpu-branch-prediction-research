#ifndef BRANCH_ANBA_NN_FROZEN_H
#define BRANCH_ANBA_NN_FROZEN_H

#include <cstdint>
#include <cstdlib>

#include "address.h"
#include "anba_int8.h"
#include "modules.h"

// Offline-trained INT8 network. Weights do not update at runtime.
class nn_frozen : champsim::modules::branch_predictor
{
  anba::Model model{};
  uint32_t ghist = 0;
  bool ready = false;

public:
  using branch_predictor::branch_predictor;

  void initialize_branch_predictor();
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
};

#endif

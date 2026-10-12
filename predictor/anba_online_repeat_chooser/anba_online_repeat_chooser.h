#ifndef BRANCH_ANBA_ONLINE_REPEAT_CHOOSER_H
#define BRANCH_ANBA_ONLINE_REPEAT_CHOOSER_H

#include <array>
#include <cstdint>

#include "address.h"
#include "modules.h"
#include "msl/fwcounter.h"

// Online perceptron trained only on bimodal mispredictions; aggregate via per-PC chooser.
class anba_online_repeat_chooser : champsim::modules::branch_predictor
{
  static constexpr std::size_t TABLE_SIZE = 16384;
  static constexpr std::size_t PRIME = 16381;
  static constexpr std::size_t BITS = 2;
  static constexpr int32_t MODEL_MARGIN = 4;
  static constexpr int32_t CHOOSER_TRUST = 4;

  std::array<champsim::msl::fwcounter<BITS>, TABLE_SIZE> table{};
  int32_t ol_w[32]{};
  int32_t ol_b = 0;
  std::array<int8_t, 16384> chooser{};
  uint32_t ghist = 0;
  bool last_uncertain = false;
  bool last_override = false;
  bool last_pred = false;
  bool last_bim_pred = false;
  bool last_model_pred = false;
  int8_t last_x[32]{};
  std::size_t last_h = 0;

  [[nodiscard]] static constexpr auto hash(champsim::address ip) { return ip.to<unsigned long>() % PRIME; }

  void train_on_mispredict(const int8_t* x, bool taken);
  void update_chooser(std::size_t h, bool bim_ok, bool model_ok);

public:
  using branch_predictor::branch_predictor;

  void initialize_branch_predictor();
  bool predict_branch(champsim::address ip);
  void last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type);
  ~anba_online_repeat_chooser();
};

#endif

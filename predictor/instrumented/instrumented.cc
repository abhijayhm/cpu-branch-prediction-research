#include "instrumented.h"

void instrumented::initialize_branch_predictor()
{
  const char* path = std::getenv("ANBA_INSTRUMENT_OUT");
  if (path != nullptr && path[0] != '\0') {
    log = std::fopen(path, "w");
    if (log != nullptr) {
      std::fprintf(log, "pc,history,taken,branch_type,predicted\n");
    }
  }
  const char* cap_env = std::getenv("ANBA_INSTRUMENT_CAP");
  if (cap_env != nullptr && cap_env[0] != '\0') {
    cap = static_cast<uint64_t>(std::strtoull(cap_env, nullptr, 10));
  }
}

bool instrumented::predict_branch(champsim::address ip)
{
  auto value = table[hash(ip)];
  return value.value() > (value.maximum / 2);
}

void instrumented::last_branch_result(champsim::address ip, champsim::address branch_target, bool taken, uint8_t branch_type)
{
  (void)branch_target;
  const bool predicted = table[hash(ip)].value() > (table[hash(ip)].maximum / 2);
  if (log != nullptr && (cap == 0 || records < cap)) {
    std::fprintf(log, "%llu,%u,%u,%u,%u\n", static_cast<unsigned long long>(ip.to<uint64_t>()), ghist, taken ? 1u : 0u,
                 static_cast<unsigned>(branch_type), predicted ? 1u : 0u);
    records++;
    if (cap != 0 && records == cap) {
      std::fflush(log);
    }
  }
  table[hash(ip)] += taken ? 1 : -1;
  ghist = ((ghist << 1) | (taken ? 1u : 0u)) & ((1u << HIST_BITS) - 1u);
}

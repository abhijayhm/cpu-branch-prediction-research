#ifndef ANBA_ONLINE_EPOCH_STATS_H
#define ANBA_ONLINE_EPOCH_STATS_H

#include <cstdio>
#include <cstdlib>

namespace anba_epoch
{

struct Counters {
  unsigned long long uncertain = 0;
  unsigned long long override_n = 0;
  unsigned long long override_correct = 0;
  unsigned long long train_updates = 0;
  int epoch = 1;
};

inline Counters& counters()
{
  static Counters c;
  return c;
}

inline void flush_on_exit()
{
  auto& c = counters();
  const double rate = c.uncertain > 0 ? static_cast<double>(c.override_n) / static_cast<double>(c.uncertain) : 0.0;
  const double prec =
      c.override_n > 0 ? static_cast<double>(c.override_correct) / static_cast<double>(c.override_n) : 0.0;
  std::fprintf(
      stderr,
      "ANBA_ONLINE_EPOCH_STATS epoch=%d train_updates=%llu uncertain=%llu override=%llu override_correct=%llu "
      "override_rate=%.4f override_precision=%.4f\n",
      c.epoch, c.train_updates, c.uncertain, c.override_n, c.override_correct, rate, prec);
  std::fflush(stderr);
}

struct AtExitRegistrar {
  AtExitRegistrar() { std::atexit(flush_on_exit); }
};

inline void ensure_registered()
{
  static AtExitRegistrar reg;
}

inline void read_epoch_from_env()
{
  if (const char* e = std::getenv("ANBA_EPOCH")) {
    counters().epoch = static_cast<int>(std::strtol(e, nullptr, 10));
    if (counters().epoch < 1) {
      counters().epoch = 1;
    }
  }
}

} // namespace anba_epoch

#endif

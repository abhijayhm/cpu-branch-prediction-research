#ifndef ANBA_GATE_STATS_H
#define ANBA_GATE_STATS_H

#include <cstdio>
#include <cstdlib>

namespace anba_gate
{

struct Counters {
  unsigned long long uncertain = 0;
  unsigned long long override_n = 0;
  unsigned long long override_correct = 0;
};

inline Counters& counters()
{
  static Counters c;
  return c;
}

inline void flush_on_exit()
{
  auto& c = counters();
  if (c.uncertain == 0) {
    return;
  }
  const double rate = static_cast<double>(c.override_n) / static_cast<double>(c.uncertain);
  const double prec = c.override_n > 0 ? static_cast<double>(c.override_correct) / static_cast<double>(c.override_n) : 0.0;
  std::fprintf(
      stderr,
      "ANBA_GATE_STATS uncertain=%llu override=%llu override_correct=%llu override_rate=%.4f override_precision=%.4f\n",
      c.uncertain, c.override_n, c.override_correct, rate, prec);
  std::fflush(stderr);
}

struct AtExitRegistrar {
  AtExitRegistrar() { std::atexit(flush_on_exit); }
};

inline void ensure_registered()
{
  static AtExitRegistrar reg;
}

} // namespace anba_gate

#endif

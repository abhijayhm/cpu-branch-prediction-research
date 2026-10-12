// One-shot logit CLI for parity: ANBA_MODEL_PATH=... echo "tok1 tok2 ..." | ./cpp_transformer_logit
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "anba_transformer.h"

int main()
{
  const char* path = std::getenv("ANBA_MODEL_PATH");
  anba_t::Model model;
  if (!model.load(path)) {
    std::fprintf(stderr, "load failed\n");
    return 1;
  }
  std::vector<uint32_t> toks;
  int x = 0;
  while (std::scanf("%d", &x) == 1) {
    toks.push_back(static_cast<uint32_t>(x));
  }
  if (toks.empty()) {
    std::fprintf(stderr, "no tokens\n");
    return 1;
  }
  const int L = static_cast<int>(std::min(toks.size(), static_cast<size_t>(model.seq_len)));
  std::printf("%.8f\n", model.forward_logit(toks.data(), L));
  return 0;
}

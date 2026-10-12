#ifndef ANBA_TRANSFORMER_H
#define ANBA_TRANSFORMER_H

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

namespace anba_t
{

constexpr uint32_t MAGIC = 0x414e4254;
constexpr uint32_t VERSION = 2;
constexpr uint32_t PC_HASH_BITS = 11;
constexpr uint32_t PC_HASH_MASK = (1u << PC_HASH_BITS) - 1u;
constexpr uint32_t VOCAB_BRANCH = (1u << PC_HASH_BITS) * 2u;
constexpr uint32_t VOCAB_CURRENT_OFFSET = VOCAB_BRANCH;
constexpr uint32_t VOCAB_SIZE = VOCAB_CURRENT_OFFSET + (1u << PC_HASH_BITS);
constexpr uint32_t SEQ_LEN = 64;

inline uint32_t pc_hash(uint64_t ip) { return static_cast<uint32_t>((ip >> 2) & PC_HASH_MASK); }

inline uint32_t branch_token(uint64_t ip, bool taken) { return (pc_hash(ip) << 1u) | (taken ? 1u : 0u); }

inline uint32_t current_token(uint64_t ip) { return VOCAB_CURRENT_OFFSET + pc_hash(ip); }

struct Tensor {
  std::vector<int32_t> shape;
  std::vector<float> data;
  const float* ptr() const { return data.data(); }
  float* ptr() { return data.data(); }
};

using TensorMap = std::unordered_map<std::string, Tensor>;

inline bool read_tensor(std::FILE* f, TensorMap& out)
{
  uint32_t name_len = 0;
  if (std::fread(&name_len, 4, 1, f) != 1 || name_len == 0 || name_len > 256) {
    return false;
  }
  std::string name(name_len, '\0');
  if (std::fread(name.data(), 1, name_len, f) != name_len) {
    return false;
  }
  uint32_t ndim = 0;
  if (std::fread(&ndim, 4, 1, f) != 1 || ndim == 0 || ndim > 8) {
    return false;
  }
  Tensor t;
  size_t n = 1;
  for (uint32_t i = 0; i < ndim; i++) {
    uint32_t d = 0;
    if (std::fread(&d, 4, 1, f) != 1) {
      return false;
    }
    t.shape.push_back(static_cast<int32_t>(d));
    n *= d;
  }
  t.data.resize(n);
  if (std::fread(t.data.data(), 4, n, f) != n) {
    return false;
  }
  out[name] = std::move(t);
  return true;
}

inline void vec_add(int n, const float* a, const float* b, float* y)
{
  for (int i = 0; i < n; i++) {
    y[i] = a[i] + b[i];
  }
}

inline void linear(const float* x, const Tensor& w, const Tensor& b, int in_f, int out_f, float* y)
{
  for (int o = 0; o < out_f; o++) {
    float s = b.data[static_cast<size_t>(o)];
    const float* row = w.ptr() + static_cast<size_t>(o) * in_f;
    for (int i = 0; i < in_f; i++) {
      s += row[i] * x[i];
    }
    y[o] = s;
  }
}

inline void layer_norm(const float* x, const Tensor& gamma, const Tensor& beta, int n, float* y)
{
  float mean = 0.f;
  for (int i = 0; i < n; i++) {
    mean += x[i];
  }
  mean /= static_cast<float>(n);
  float var = 0.f;
  for (int i = 0; i < n; i++) {
    const float d = x[i] - mean;
    var += d * d;
  }
  var /= static_cast<float>(n);
  const float inv = 1.f / std::sqrt(var + 1e-5f);
  for (int i = 0; i < n; i++) {
    y[i] = (x[i] - mean) * inv * gamma.data[static_cast<size_t>(i)] + beta.data[static_cast<size_t>(i)];
  }
}

inline float gelu(float x)
{
  return 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

inline void encoder_layer(
    int L, int d_model, int nhead, const TensorMap& tensors, const std::string& pfx, std::vector<float>& x,
    std::vector<float>& n1, std::vector<float>& qkv_all, std::vector<float>& attn_out, std::vector<float>& scores,
    std::vector<float>& proj, std::vector<float>& tmp, std::vector<float>& ff)
{
  const int dm = d_model;
  const int head_dim = dm / nhead;
  const auto& n1w = tensors.at(pfx + "norm1.weight");
  const auto& n1b = tensors.at(pfx + "norm1.bias");
  const auto& n2w = tensors.at(pfx + "norm2.weight");
  const auto& n2b = tensors.at(pfx + "norm2.bias");
  const auto& in_w = tensors.at(pfx + "self_attn.in_proj_weight");
  const auto& in_b = tensors.at(pfx + "self_attn.in_proj_bias");
  const auto& out_w = tensors.at(pfx + "self_attn.out_proj.weight");
  const auto& out_b = tensors.at(pfx + "self_attn.out_proj.bias");
  const auto& l1w = tensors.at(pfx + "linear1.weight");
  const auto& l1b = tensors.at(pfx + "linear1.bias");
  const auto& l2w = tensors.at(pfx + "linear2.weight");
  const auto& l2b = tensors.at(pfx + "linear2.bias");
  const int ff_dim = l1w.shape[0];

  n1.resize(static_cast<size_t>(L) * dm);
  qkv_all.resize(static_cast<size_t>(L) * 3 * dm);
  attn_out.resize(static_cast<size_t>(L) * dm);
  proj.resize(dm);
  scores.resize(L);
  tmp.resize(dm);
  ff.resize(static_cast<size_t>(ff_dim));

  for (int t = 0; t < L; t++) {
    layer_norm(x.data() + static_cast<size_t>(t) * dm, n1w, n1b, dm, n1.data() + static_cast<size_t>(t) * dm);
    linear(n1.data() + static_cast<size_t>(t) * dm, in_w, in_b, dm, 3 * dm,
           qkv_all.data() + static_cast<size_t>(t) * 3 * dm);
  }

  const float inv_sqrt = 1.f / std::sqrt(static_cast<float>(head_dim));
  for (int t = 0; t < L; t++) {
    for (int h = 0; h < nhead; h++) {
      const float* q = qkv_all.data() + static_cast<size_t>(t) * 3 * dm + h * head_dim;
      float max_logit = -1e9f;
      for (int j = 0; j < L; j++) {
        const float* k = qkv_all.data() + static_cast<size_t>(j) * 3 * dm + dm + h * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; d++) {
          dot += q[d] * k[d];
        }
        dot *= inv_sqrt;
        if (j > t) {
          dot = -1e9f;
        }
        scores[j] = dot;
        max_logit = std::max(max_logit, dot);
      }
      float sum = 0.f;
      for (int j = 0; j < L; j++) {
        scores[j] = std::exp(scores[j] - max_logit);
        sum += scores[j];
      }
      for (int j = 0; j < L; j++) {
        scores[j] /= sum;
      }
      float* out_h = attn_out.data() + static_cast<size_t>(t) * dm + h * head_dim;
      for (int d = 0; d < head_dim; d++) {
        out_h[d] = 0.f;
      }
      for (int j = 0; j < L; j++) {
        const float* v = qkv_all.data() + static_cast<size_t>(j) * 3 * dm + 2 * dm + h * head_dim;
        for (int d = 0; d < head_dim; d++) {
          out_h[d] += scores[j] * v[d];
        }
      }
    }
    linear(attn_out.data() + static_cast<size_t>(t) * dm, out_w, out_b, dm, dm, proj.data());
    vec_add(dm, x.data() + static_cast<size_t>(t) * dm, proj.data(), x.data() + static_cast<size_t>(t) * dm);
  }

  for (int t = 0; t < L; t++) {
    layer_norm(x.data() + static_cast<size_t>(t) * dm, n2w, n2b, dm, tmp.data());
    linear(tmp.data(), l1w, l1b, dm, ff_dim, ff.data());
    for (int i = 0; i < ff_dim; i++) {
      ff[static_cast<size_t>(i)] = gelu(ff[static_cast<size_t>(i)]);
    }
    linear(ff.data(), l2w, l2b, ff_dim, dm, tmp.data());
    vec_add(dm, x.data() + static_cast<size_t>(t) * dm, tmp.data(), x.data() + static_cast<size_t>(t) * dm);
  }
}

struct Model {
  TensorMap tensors;
  uint32_t seq_len = SEQ_LEN;
  uint32_t d_model = 48;
  uint32_t nhead = 4;
  uint32_t nlayers = 2;
  float margin = 1.f;
  bool loaded = false;
  mutable std::vector<float> work_x;
  mutable std::vector<float> work_n1;
  mutable std::vector<float> work_qkv;
  mutable std::vector<float> work_attn;
  mutable std::vector<float> work_scores;
  mutable std::vector<float> work_proj;
  mutable std::vector<float> work_tmp;
  mutable std::vector<float> work_ff;

  bool load(const char* path)
  {
    if (path == nullptr || path[0] == '\0') {
      return false;
    }
    std::FILE* f = std::fopen(path, "rb");
    if (f == nullptr) {
      return false;
    }
    uint32_t magic = 0, ver = 0, arch = 0, vocab = 0, nt = 0;
    if (std::fread(&magic, 4, 1, f) != 1 || magic != MAGIC || std::fread(&ver, 4, 1, f) != 1 || ver != VERSION
        || std::fread(&arch, 4, 1, f) != 1 || std::fread(&seq_len, 4, 1, f) != 1 || std::fread(&d_model, 4, 1, f) != 1
        || std::fread(&nhead, 4, 1, f) != 1 || std::fread(&nlayers, 4, 1, f) != 1 || std::fread(&vocab, 4, 1, f) != 1
        || std::fread(&margin, 4, 1, f) != 1 || std::fread(&nt, 4, 1, f) != 1) {
      std::fclose(f);
      return false;
    }
    tensors.clear();
    for (uint32_t i = 0; i < nt; i++) {
      if (!read_tensor(f, tensors)) {
        std::fclose(f);
        return false;
      }
    }
    std::fclose(f);
    loaded = tensors.count("embed.weight") > 0;
    return loaded;
  }

  float forward_logit(const uint32_t* tokens, int L) const
  {
    if (!loaded || L <= 0) {
      return 0.f;
    }
    L = std::min(L, static_cast<int>(seq_len));
    const int dm = static_cast<int>(d_model);
    const auto& embed = tensors.at("embed.weight");
    const auto& pos_embed = tensors.at("pos_embed.weight");
    work_x.resize(static_cast<size_t>(L) * dm);
    for (int t = 0; t < L; t++) {
      const uint32_t tok = std::min(tokens[t], VOCAB_SIZE - 1);
      const float* erow = embed.ptr() + static_cast<size_t>(tok) * dm;
      const float* prow = pos_embed.ptr() + static_cast<size_t>(t) * dm;
      for (int d = 0; d < dm; d++) {
        work_x[static_cast<size_t>(t) * dm + d] = erow[d] + prow[d];
      }
    }
    for (uint32_t layer = 0; layer < nlayers; layer++) {
      const std::string pfx = "encoder.layers." + std::to_string(layer) + ".";
      encoder_layer(L, dm, static_cast<int>(nhead), tensors, pfx, work_x, work_n1, work_qkv, work_attn, work_scores,
                    work_proj, work_tmp, work_ff);
    }
    const auto& hw = tensors.at("head.weight");
    const auto& hb = tensors.at("head.bias");
    float logit = 0.f;
    linear(work_x.data() + static_cast<size_t>(L - 1) * dm, hw, hb, dm, 1, &logit);
    return logit;
  }

  bool predict(const uint32_t* tokens, int L) const { return forward_logit(tokens, L) >= 0.f; }
};

} // namespace anba_t

#endif

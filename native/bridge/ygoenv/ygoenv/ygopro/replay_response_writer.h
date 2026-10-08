#ifndef YGOENV_REPLAY_RESPONSE_WRITER_H_
#define YGOENV_REPLAY_RESPONSE_WRITER_H_

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace ygopro {

// A response may be overwritten before process() consumes it (native auto-pass
// followed by WindBot). Persist only the consumed, accepted response.
class ReplayResponseWriter {
 public:
  static void Flush(FILE* stream) {
    if (!stream || fflush(stream) != 0)
      throw std::runtime_error("Failed to flush replay stream");
  }

  void Clear() { pending_.clear(); }

  void Set(const uint8_t* data, size_t length) {
    if (length == 0 || length > 255)
      throw std::runtime_error("Invalid replay response length");
    pending_.assign(data, data + length);
  }

  void Commit(FILE* stream, bool retry) {
    if (pending_.empty())
      return;
    if (!retry) {
      const uint8_t length = static_cast<uint8_t>(pending_.size());
      if (!stream || fwrite(&length, 1, 1, stream) != 1 ||
          fwrite(pending_.data(), 1, pending_.size(), stream) != pending_.size())
        throw std::runtime_error("Failed to write replay response");
    }
    pending_.clear();
  }

 private:
  std::vector<uint8_t> pending_;
};

}  // namespace ygopro
#endif

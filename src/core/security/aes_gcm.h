#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "pomai/status.h"

namespace pomai::core {

class AesGcm {
public:
    static Status ParseKeyHex(const std::string& hex, std::array<std::uint8_t, 32>* out);
    static Status Encrypt(const std::array<std::uint8_t, 32>& key,
                          const std::array<std::uint8_t, 12>& nonce,
                          const std::vector<std::uint8_t>& plaintext,
                          std::vector<std::uint8_t>* ciphertext,
                          std::array<std::uint8_t, 16>* tag);
    static Status Decrypt(const std::array<std::uint8_t, 32>& key,
                          const std::array<std::uint8_t, 12>& nonce,
                          const std::vector<std::uint8_t>& ciphertext,
                          const std::array<std::uint8_t, 16>& tag,
                          std::vector<std::uint8_t>* plaintext);
};

} // namespace pomai::core


#pragma once

#include <array>
#include <cstdint>

namespace pomai::core {

class KeyManager {
public:
    static KeyManager& Global() {
        static KeyManager km;
        return km;
    }

    void SetKey(const std::array<std::uint8_t, 32>& key) {
        key_ = key;
        armed_ = true;
    }

    bool IsArmed() const { return armed_; }

    void Wipe() {
        for (auto& b : key_) b = 0;
        armed_ = false;
    }

private:
    KeyManager() = default;
    std::array<std::uint8_t, 32> key_{};
    bool armed_ = false;
};

} // namespace pomai::core


#pragma once

#include <cstdint>
#include <unordered_map>

#include "pomai/types.h"

namespace pomai::core {

enum class DataTemperature : uint8_t {
    kHot = 0,
    kWarm = 1,
    kCold = 2,
};

class SemanticLifecycle {
public:
    explicit SemanticLifecycle(std::size_t max_entries = 20000) : max_entries_(max_entries) {}
    void SetMaxEntries(std::size_t max_entries) { max_entries_ = max_entries; }
    void OnRead(VectorId id);
    void OnWrite(VectorId id);
    void OnDelete(VectorId id);
    DataTemperature Classify(VectorId id) const;
    std::size_t CountHot() const;
    std::size_t CountWarm() const;
    std::size_t CountCold() const;

private:
    struct Entry {
        std::uint32_t reads = 0;
        std::uint32_t writes = 0;
    };
    void EvictIfNeeded();
    std::unordered_map<VectorId, Entry> table_;
    std::size_t max_entries_ = 20000;
};

} // namespace pomai::core


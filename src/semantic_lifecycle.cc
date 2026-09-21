#include "semantic_lifecycle.h"

namespace pomai::core {

void SemanticLifecycle::OnRead(VectorId id) {
    if (max_entries_ == 0) return;
    auto* entry = table_.Find(id);
    if (entry) {
        entry->reads++;
        return;
    }
    EvictIfNeeded();
    table_.Put(id, Entry{.reads = 1, .writes = 0});
}

void SemanticLifecycle::OnWrite(VectorId id) {
    if (max_entries_ == 0) return;
    auto* entry = table_.Find(id);
    if (entry) {
        entry->writes++;
        return;
    }
    EvictIfNeeded();
    table_.Put(id, Entry{.reads = 0, .writes = 1});
}

void SemanticLifecycle::OnDelete(VectorId id) {
    table_.Erase(id);
}

DataTemperature SemanticLifecycle::Classify(VectorId id) const {
    const auto* entry = table_.Find(id);
    if (!entry) return DataTemperature::kCold;
    const auto score = entry->reads * 3 + entry->writes;
    if (score >= 20) return DataTemperature::kHot;
    if (score >= 5) return DataTemperature::kWarm;
    return DataTemperature::kCold;
}

std::size_t SemanticLifecycle::CountHot() const {
    std::size_t n = 0;
    table_.ForEach([&](VectorId id, const Entry&) {
        if (Classify(id) == DataTemperature::kHot) ++n;
    });
    return n;
}

std::size_t SemanticLifecycle::CountWarm() const {
    std::size_t n = 0;
    table_.ForEach([&](VectorId id, const Entry&) {
        if (Classify(id) == DataTemperature::kWarm) ++n;
    });
    return n;
}

std::size_t SemanticLifecycle::CountCold() const {
    std::size_t n = 0;
    table_.ForEach([&](VectorId id, const Entry&) {
        if (Classify(id) == DataTemperature::kCold) ++n;
    });
    return n;
}

void SemanticLifecycle::EvictIfNeeded() {
    if (max_entries_ == 0) return;
    while (table_.size() >= max_entries_) {
        VectorId victim = 0;
        if (!table_.FindAny(&victim)) break;
        if (!table_.Erase(victim)) break;
    }
}

} // namespace pomai::core


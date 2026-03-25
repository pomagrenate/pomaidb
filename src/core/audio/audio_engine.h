#pragma once

#include <cstdint>
#include <functional>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "pomai/status.h"

namespace pomai::core {

/// A single stored audio frame (one embedding window).
struct AudioFrame {
    uint64_t timestamp;       // Frame start time in milliseconds
    std::vector<float> embedding;
};

/// A search result entry from AudioEngine::Search.
struct AudioHit {
    uint64_t clip_id;
    uint64_t timestamp;
    float    distance;
};

/**
 * AudioEngine — frame-aligned audio embedding storage.
 *
 * Each clip is a sequence of (timestamp, embedding) frames.
 * Search performs brute-force L2 over all frames in an optional time range,
 * appropriate for short clip libraries on edge devices (Jetson Nano, RPi).
 *
 * Thread safety: single-writer, single-reader (VectorRuntime guarantees this).
 */
class AudioEngine {
public:
    AudioEngine(std::string path, std::size_t max_frames_total);

    Status Open();
    Status Close();

    /// Store one embedding frame for clip_id at the given timestamp (ms).
    /// Evicts oldest frame across all clips if max_frames_total is reached.
    Status Put(uint64_t clip_id, uint64_t timestamp_ms, std::span<const float> embedding);

    /// Delete all frames for a clip.
    Status Delete(uint64_t clip_id);

    /// Return all frames for a clip (sorted by timestamp).
    Status GetFrames(uint64_t clip_id, std::vector<AudioFrame>* out) const;

    /**
     * Nearest-neighbor search over stored frames.
     * @param query         Query embedding (must match stored dim).
     * @param time_start_ms Start of time window (0 = no lower bound).
     * @param time_end_ms   End of time window (0 = no upper bound).
     * @param topk          Max results.
     * @param out           Output hits sorted by ascending L2 distance.
     */
    Status Search(std::span<const float> query,
                  uint64_t               time_start_ms,
                  uint64_t               time_end_ms,
                  uint32_t               topk,
                  std::vector<AudioHit>* out) const;

    /// Iterate all clips with a callback.
    void ForEach(const std::function<void(uint64_t clip_id, const AudioFrame&)>& fn) const;

    std::size_t FrameCount() const noexcept { return total_frames_; }

private:
    std::string path_;
    std::size_t max_frames_total_;
    std::size_t total_frames_{0};

    // clip_id → ordered list of frames (kept sorted by timestamp on insert)
    std::unordered_map<uint64_t, std::vector<AudioFrame>> clips_;
};

} // namespace pomai::core

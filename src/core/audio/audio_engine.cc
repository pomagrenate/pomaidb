#include "core/audio/audio_engine.h"

#include <algorithm>
#include <cmath>

namespace pomai::core {

AudioEngine::AudioEngine(std::string path, std::size_t max_frames_total)
    : path_(std::move(path)), max_frames_total_(max_frames_total)
{}

Status AudioEngine::Open()  { return Status::Ok(); }
Status AudioEngine::Close() { return Status::Ok(); }

Status AudioEngine::Put(uint64_t clip_id, uint64_t timestamp_ms,
                        std::span<const float> embedding)
{
    if (embedding.empty())
        return Status::InvalidArgument("audio embedding must be non-empty");

    // Evict the oldest frame (globally) when at capacity.
    if (total_frames_ >= max_frames_total_ && max_frames_total_ > 0) {
        // Find the clip with the oldest (smallest timestamp) frame and drop it.
        uint64_t oldest_clip = 0;
        uint64_t oldest_ts   = UINT64_MAX;
        for (auto& [cid, frames] : clips_) {
            if (!frames.empty() && frames.front().timestamp < oldest_ts) {
                oldest_ts   = frames.front().timestamp;
                oldest_clip = cid;
            }
        }
        auto it = clips_.find(oldest_clip);
        if (it != clips_.end() && !it->second.empty()) {
            it->second.erase(it->second.begin());
            --total_frames_;
            if (it->second.empty())
                clips_.erase(it);
        }
    }

    auto& frames = clips_[clip_id];
    AudioFrame f;
    f.timestamp = timestamp_ms;
    f.embedding.assign(embedding.begin(), embedding.end());

    // Insert in timestamp order.
    auto pos = std::lower_bound(frames.begin(), frames.end(), f,
        [](const AudioFrame& a, const AudioFrame& b) {
            return a.timestamp < b.timestamp;
        });
    frames.insert(pos, std::move(f));
    ++total_frames_;
    return Status::Ok();
}

Status AudioEngine::Delete(uint64_t clip_id)
{
    auto it = clips_.find(clip_id);
    if (it == clips_.end())
        return Status::NotFound("audio clip not found");
    total_frames_ -= it->second.size();
    clips_.erase(it);
    return Status::Ok();
}

Status AudioEngine::GetFrames(uint64_t clip_id, std::vector<AudioFrame>* out) const
{
    if (!out) return Status::InvalidArgument("out is null");
    auto it = clips_.find(clip_id);
    if (it == clips_.end())
        return Status::NotFound("audio clip not found");
    *out = it->second;
    return Status::Ok();
}

Status AudioEngine::Search(std::span<const float> query,
                           uint64_t               time_start_ms,
                           uint64_t               time_end_ms,
                           uint32_t               topk,
                           std::vector<AudioHit>* out) const
{
    if (!out)   return Status::InvalidArgument("out is null");
    if (topk == 0) { out->clear(); return Status::Ok(); }
    if (query.empty()) return Status::InvalidArgument("query embedding must be non-empty");

    const bool has_lower = (time_start_ms != 0);
    const bool has_upper = (time_end_ms   != 0);

    // Brute-force L2 scan over all frames in the time window.
    std::vector<AudioHit> candidates;
    for (const auto& [clip_id, frames] : clips_) {
        for (const auto& frame : frames) {
            if (has_lower && frame.timestamp < time_start_ms) continue;
            if (has_upper && frame.timestamp > time_end_ms)   continue;
            if (frame.embedding.size() != query.size())       continue;

            float dist = 0.0f;
            for (std::size_t i = 0; i < query.size(); ++i) {
                float d = query[i] - frame.embedding[i];
                dist += d * d;
            }
            candidates.push_back(AudioHit{clip_id, frame.timestamp, dist});
        }
    }

    const std::size_t k = std::min<std::size_t>(topk, candidates.size());
    std::partial_sort(candidates.begin(), candidates.begin() + static_cast<std::ptrdiff_t>(k),
                      candidates.end(),
                      [](const AudioHit& a, const AudioHit& b) { return a.distance < b.distance; });
    candidates.resize(k);
    *out = std::move(candidates);
    return Status::Ok();
}

void AudioEngine::ForEach(const std::function<void(uint64_t, const AudioFrame&)>& fn) const
{
    for (const auto& [clip_id, frames] : clips_)
        for (const auto& f : frames)
            fn(clip_id, f);
}

} // namespace pomai::core
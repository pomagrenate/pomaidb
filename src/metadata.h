#pragma once
#include <cstdint>
#include "types.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace pomai
{
    /**
     * @brief Metadata associated with a vector.
     */
    struct Metadata
    {
        std::string tenant;      // Multi-tenancy filtering
        std::string device_id;   // Partition key
        std::string location_id; // Partition key
        uint64_t timestamp = 0;  // Timestamp for temporal queries
        uint64_t lsn = 0;        // Write sequence number for time-travel queries
        std::string payload;     // Arbitrary Document/JSON vector payload

        Metadata() = default;
        explicit Metadata(std::string t, uint64_t ts = 0, std::string p = "")
            : tenant(std::move(t)), timestamp(ts), payload(std::move(p)) {}

        bool operator==(const Metadata& other) const {
            return tenant == other.tenant && device_id == other.device_id && location_id == other.location_id &&
                   timestamp == other.timestamp && lsn == other.lsn && payload == other.payload;
        }
    };

    struct Filter
    {
        std::string field;
        std::string value;

        // Temporal Range
        uint64_t min_ts = 0;
        uint64_t max_ts = 0;

        Filter() = default;
        Filter(std::string f, std::string v)
            : field(std::move(f)), value(std::move(v)) {}

        static Filter TimeRange(uint64_t min, uint64_t max) {
            Filter f;
            f.field = "timestamp";
            f.min_ts = min;
            f.max_ts = max;
            return f;
        }

        bool Matches(const Metadata& meta) const {
            if (field == "tenant" && meta.tenant != value) return false;
            if (field == "device_id" && meta.device_id != value) return false;
            if (field == "location_id" && meta.location_id != value) return false;

            if (field == "timestamp") {
                if (min_ts > 0 && meta.timestamp < min_ts) return false;
                if (max_ts > 0 && meta.timestamp > max_ts) return false;
            }

            return true;
        }
    };
    
    struct SearchOptions
    {
        std::vector<Filter> filters;
        uint64_t as_of_ts = 0;   // 0 = latest
        uint64_t as_of_lsn = 0;  // 0 = latest
        std::string partition_device_id;
        std::string partition_location_id;
        bool force_fanout = false;
        uint32_t routing_probe_override = 0;
        bool zero_copy = false;
        
        SearchOptions() = default;
        
        bool Matches(const Metadata& meta) const {
            if (as_of_ts > 0 && meta.timestamp > as_of_ts) return false;
            if (as_of_lsn > 0 && meta.lsn > as_of_lsn) return false;
            if (!partition_device_id.empty() && meta.device_id != partition_device_id) return false;
            if (!partition_location_id.empty() && meta.location_id != partition_location_id) return false;
            for (const auto& filter : filters) {
                if (!filter.Matches(meta)) {
                    return false;
                }
            }
            return true;
        }
    };

} // namespace pomai

#pragma once

#include <string>
#include <unordered_map>

#include "pomai/status.h"

namespace pomai::core {

class KeyValueEngine {
public:
    KeyValueEngine(std::string path, std::size_t max_entries);
    Status Open();
    Status Close();
    Status Put(std::string_view key, std::string_view value);
    Status Get(std::string_view key, std::string* out) const;
    Status Delete(std::string_view key);

private:
    std::string path_;
    std::size_t max_entries_;
    std::unordered_map<std::string, std::string> kv_;
};

} // namespace pomai::core


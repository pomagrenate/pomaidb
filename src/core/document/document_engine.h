#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "core/query/lexical_index.h"
#include "pomai/status.h"

namespace pomai::core {

/// A search hit returned by DocumentEngine::Search.
struct DocumentHit {
    uint64_t doc_id;
    float    score;   // BM25 relevance score
};

/**
 * DocumentEngine — JSON document store with BM25 full-text search.
 *
 * Documents are stored as raw JSON strings and indexed via LexicalIndex
 * (BM25) for keyword search.  Field-path filtering is left to the caller
 * (parse the returned JSON string against filter_expression).
 *
 * Edge rationale: closes the gap where MultiModalQuery::filter_expression
 * is declared but unimplemented. Suitable for multi-tenant deployments where
 * per-device metadata (sensor config, labels) needs structured lookup.
 *
 * Thread safety: single-writer, single-reader.
 */
class DocumentEngine {
public:
    DocumentEngine(std::string path, std::size_t max_documents);

    Status Open();
    Status Close();

    /// Store or replace a document.  The content is indexed for full-text search.
    Status Put(uint64_t doc_id, std::string_view json_content);

    /// Retrieve raw JSON content for a document.
    Status Get(uint64_t doc_id, std::string* out) const;

    /// Remove a document and de-index it.
    Status Delete(uint64_t doc_id);

    /// Check existence without fetching content.
    bool Exists(uint64_t doc_id) const;

    /**
     * Full-text BM25 search over all stored documents.
     * @param query  Space-separated keywords.
     * @param topk   Max results.
     * @param out    Hits sorted by descending BM25 score.
     */
    Status Search(const std::string& query,
                  uint32_t           topk,
                  std::vector<DocumentHit>* out);

    std::size_t Count() const noexcept { return docs_.size(); }

    void ForEach(const std::function<void(uint64_t doc_id, std::string_view content)>& fn) const;

private:
    std::string path_;
    std::size_t max_documents_;

    std::unordered_map<uint64_t, std::string> docs_;
    LexicalIndex index_;
};

} // namespace pomai::core
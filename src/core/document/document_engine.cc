#include "core/document/document_engine.h"

#include <algorithm>

namespace pomai::core {

DocumentEngine::DocumentEngine(std::string path, std::size_t max_documents)
    : path_(std::move(path)), max_documents_(max_documents)
{}

Status DocumentEngine::Open()  { return Status::Ok(); }
Status DocumentEngine::Close() { return Status::Ok(); }

Status DocumentEngine::Put(uint64_t doc_id, std::string_view json_content)
{
    if (json_content.empty())
        return Status::InvalidArgument("document content must be non-empty");

    // If replacing an existing document, de-index old content first.
    auto it = docs_.find(doc_id);
    if (it != docs_.end()) {
        index_.Remove(static_cast<VectorId>(doc_id));
        it->second = std::string(json_content);
    } else {
        // Enforce capacity: evict the entry with the smallest id (arbitrary but stable).
        if (docs_.size() >= max_documents_ && max_documents_ > 0) {
            auto oldest = std::min_element(docs_.begin(), docs_.end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            if (oldest != docs_.end()) {
                index_.Remove(static_cast<VectorId>(oldest->first));
                docs_.erase(oldest);
            }
        }
        docs_.emplace(doc_id, std::string(json_content));
    }

    // Index the full JSON text for keyword search.
    index_.Add(static_cast<VectorId>(doc_id), std::string(json_content));
    return Status::Ok();
}

Status DocumentEngine::Get(uint64_t doc_id, std::string* out) const
{
    if (!out) return Status::InvalidArgument("out is null");
    auto it = docs_.find(doc_id);
    if (it == docs_.end()) return Status::NotFound("document not found");
    *out = it->second;
    return Status::Ok();
}

Status DocumentEngine::Delete(uint64_t doc_id)
{
    auto it = docs_.find(doc_id);
    if (it == docs_.end()) return Status::NotFound("document not found");
    index_.Remove(static_cast<VectorId>(doc_id));
    docs_.erase(it);
    return Status::Ok();
}

bool DocumentEngine::Exists(uint64_t doc_id) const
{
    return docs_.count(doc_id) > 0;
}

Status DocumentEngine::Search(const std::string&        query,
                               uint32_t                  topk,
                               std::vector<DocumentHit>* out)
{
    if (!out)   return Status::InvalidArgument("out is null");
    if (topk == 0) { out->clear(); return Status::Ok(); }

    std::vector<LexicalHit> hits;
    index_.Search(query, topk, &hits);

    out->clear();
    out->reserve(hits.size());
    for (const auto& h : hits)
        out->push_back(DocumentHit{static_cast<uint64_t>(h.id), h.score});
    return Status::Ok();
}

void DocumentEngine::ForEach(
    const std::function<void(uint64_t, std::string_view)>& fn) const
{
    for (const auto& [doc_id, content] : docs_)
        fn(doc_id, content);
}

} // namespace pomai::core
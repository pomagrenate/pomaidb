# PomaiDB — Production HNSW Persistence Specification

## 1. Container Overview: The `.pom` Locule Architecture

PomaiDB stores spatial partitions as immutable `.pom` (Pomegranate Object Model) container files. Each `.pom` container represents a **Locule**—a spatially bounded region in vector space defined by a centroid coordinate and a bounding radius.

A Locule container packages one or more immutable **Arils** (autonomous sub-segments) along with an embedded spatial footer and a 64-bit aligned directory table. Within each Aril, an upstream-standard HNSW graph is embedded and persisted directly alongside the quantized **Pulp** (SQ8) representation and the exact **Seed Kernel** (FP32) vector pool.

---

## 2. Byte-Level Locule Container Layout

A `.pom` container file adheres to 64-byte cache-line alignment across all internal block offsets.

```text
+-------------------------------------------------------------------------+
| PomaiFileHeader (48 bytes, aligned to 64 bytes)                         |
|   magic: 0x504F4D41 ('POMA'), version: 1, kind: 1 (Locule)              |
|   generation: uint64, dimension: uint32, aril_count: uint32              |
|   directory_offset: uint64, directory_size: uint64, footer_offset: uint64|
|   checksum: uint32 (CRC32C of Aril Directory table)                     |
+-------------------------------------------------------------------------+
| Aril Block 0 (64-byte aligned)                                          |
|   ArilHeader + Pulp Block + Seed Kernel + Graph Block + Scar + Dir + Meta|
+-------------------------------------------------------------------------+
| Aril Block 1 ... N (64-byte aligned)                                    |
+-------------------------------------------------------------------------+
| Aril Directory Table                                                    |
|   Array of ArilDirectoryEntry [0 .. aril_count - 1] (32 bytes each)      |
+-------------------------------------------------------------------------+
| Locule Footer                                                           |
|   LoculeFooter (24 bytes) + Centroid FP32 Array (dim * sizeof(float))   |
+-------------------------------------------------------------------------+
```

### 2.1 Container Header (`PomaiFileHeader`, 48 bytes)

| Field | Type | Offset | Description |
|---|---|---|---|
| `magic` | `uint32_t` | `0x00` | Magic signature: `0x504F4D41` (ASCII `"POMA"`) |
| `version` | `uint16_t` | `0x04` | Format version: `1` |
| `kind` | `uint16_t` | `0x06` | Container kind: `1` (`kContainerKindLocule`) |
| `generation` | `uint64_t` | `0x08` | Compaction generation timestamp / sequence |
| `dimension` | `uint32_t` | `0x10` | Vector space dimensionality $D$ |
| `aril_count` | `uint32_t` | `0x14` | Number of Arils packaged in this Locule |
| `directory_offset`| `uint64_t` | `0x18` | Absolute file offset to `ArilDirectoryEntry` array |
| `directory_size` | `uint64_t` | `0x20` | Size in bytes of the directory table |
| `footer_offset` | `uint64_t` | `0x28` | Absolute file offset to `LoculeFooter` |
| `checksum` | `uint32_t` | `0x30` | CRC32C checksum of the entire directory table |

### 2.2 Aril Directory Entry (`ArilDirectoryEntry`, 32 bytes)

Each Aril within the container is indexed by a directory entry:

| Field | Type | Offset | Description |
|---|---|---|---|
| `aril_id` | `uint32_t` | `0x00` | 1-based local index of the Aril within the Locule |
| `vector_count` | `uint32_t` | `0x04` | Total number of vector slots in this Aril |
| `aril_offset` | `uint64_t` | `0x08` | Absolute file offset where Aril begins |
| `aril_size` | `uint64_t` | `0x10` | Total byte size of the complete Aril payload |
| `checksum` | `uint32_t` | `0x18` | CRC32C checksum of all bytes in `[aril_offset, aril_offset + aril_size)` |
| `reserved` | `uint32_t` | `0x1C` | Padding for 8-byte alignment |

---

## 3. Aril Internal Binary Layout

An Aril is completely self-contained and relocatable. All internal offsets stored in the `ArilHeader` are relative to the beginning of the Aril (`aril_offset`).

```text
+-------------------------------------------------------------------------+
| ArilHeader (104 bytes, 64-byte aligned)                                 |
|   magic: 0x4152494C ('ARIL'), version: 1, vector_count, dimension       |
|   pulp_offset, pulp_size, quant_min, quant_inv_scale, quant_type (SQ8)  |
|   kernel_offset, kernel_size (FP32 Seed Kernel vectors)                 |
|   graph_offset, graph_size (HNSW Graph Block)                           |
|   scar_offset, scar_size (Tombstone Bitset)                             |
|   dir_offset, dir_size (VectorId -> Slot Directory)                     |
|   metadata_offset, metadata_size (Variable payload blob)                |
|   checksum (CRC32C of Aril payload)                                     |
+-------------------------------------------------------------------------+
| Pulp SQ8 Block (64-byte aligned)                                        |
+-------------------------------------------------------------------------+
| Seed Kernel FP32 Block (64-byte aligned)                                |
+-------------------------------------------------------------------------+
| Aril Graph Block (64-byte aligned)                                      |
|   +-------------------------------------------------------------------+ |
|   | PomaiHnswHeader (32 bytes)                                        | |
|   |   magic: 0x484E5357 ('HNSW'), version: 1, flags: 0                | |
|   |   checksum: CRC32C of upstream graph bytes                        | |
|   |   graph_payload_size: uint64, max_elements: uint32, num_elem: u32 | |
|   |   reserved: uint32                                                | |
|   +-------------------------------------------------------------------+ |
|   | Serialized upstream hnswlib::HierarchicalNSW binary stream payload | |
|   +-------------------------------------------------------------------+ |
+-------------------------------------------------------------------------+
| Seed Scar Tombstone Bitset Block (64-byte aligned)                      |
+-------------------------------------------------------------------------+
| Seed Directory Table (64-byte aligned)                                  |
+-------------------------------------------------------------------------+
| Metadata String Blob Block (64-byte aligned)                            |
+-------------------------------------------------------------------------+
```

---

## 4. Aril Graph Header Specification (`PomaiHnswHeader`)

The embedded HNSW graph is prefixed with a 32-byte header `PomaiHnswHeader`:

```cpp
#pragma pack(push, 1)
struct PomaiHnswHeader {
    uint32_t magic;               // 0x484E5357 ('HNSW' in little-endian)
    uint16_t version;             // 1
    uint16_t flags;               // Reserved (0)
    uint32_t checksum;            // CRC32C of the raw upstream graph payload
    uint64_t graph_payload_size;  // Byte length of upstream serialized graph
    uint32_t max_elements;        // Maximum element capacity of HNSW index
    uint32_t num_elements;        // Active vector count indexed in HNSW
    uint32_t reserved;            // Alignment padding (0)
};
#pragma pack(pop)
```

| Field | Type | Value / Meaning |
|---|---|---|
| `magic` | `uint32_t` | Fixed constant `0x484E5357` (ASCII `"HNSW"`) |
| `version` | `uint16_t` | Binary layout version `1` |
| `flags` | `uint16_t` | Bit flags for future extensions (currently `0`) |
| `checksum` | `uint32_t` | Castagnoli CRC32C over the graph payload bytes |
| `graph_payload_size` | `uint64_t` | Byte length of the raw `hnswlib` index stream |
| `max_elements` | `uint32_t` | Preallocated capacity allocated during graph building |
| `num_elements` | `uint32_t` | Exact number of vectors inserted into the graph |
| `reserved` | `uint32_t` | Pad to 32 bytes |

### 4.1 Upstream `hnswlib` Stream Format Compatibility

Immediately following `PomaiHnswHeader` at relative offset `sizeof(PomaiHnswHeader) = 32`, the exact binary byte-stream produced by upstream `hnswlib::HierarchicalNSW::saveIndex` is stored without alteration:
- `offsetLevel0_`: Stream offset of the base layer graph
- `max_elements_`: Maximum element capacity
- `cur_element_count`: Total indexed items
- `size_data_per_element_`: Byte size per vector entry
- `label_offset_`: Offset of user ID / label
- `offsetData_`: Vector data array offset
- `maxlevel_`: Top layer in the skip-list hierarchy
- `enterpoint_node_`: Global entry point node ID
- Links, levels, and neighbor adjacency lists per node

When opening the `.pom` file via zero-copy `mmap`, PomaiDB wraps the sub-buffer `[base_addr + graph_offset + 32, ...]` in a seekable `imemstream` that provides standard `std::istream` seeking semantics so `hnswlib::loadIndexNoExceptions` initializes the graph structures directly from memory without copying.

---

## 5. Corruption Detection and Fault-Tolerant Fallback

PomaiDB enforces three independent tiers of CRC32C validation:

1. **Locule Directory Integrity**: `PomaiFileHeader.checksum` validates that the list of Arils and their offsets are not corrupted.
2. **Aril Block Integrity**: `ArilDirectoryEntry.checksum` validates that the entire Aril byte range on disk is intact.
3. **Graph Payload Integrity**: `PomaiHnswHeader.checksum` validates the cryptographic integrity of the HNSW graph payload before passing it to `hnswlib`.

### Graceful Fallback Semantics

If the HNSW graph header magic `0x484E5357` is invalid or `PomaiHnswHeader.checksum` fails:
- The database **does NOT crash** or abort.
- `ArilReader::OpenFromMemory` logs a detailed warning: `[WARN] Aril {id} HNSW graph verification failed: {reason}`.
- `reader->graph_` is set to `nullptr` (`aril->HasGraph() == false`).
- Production query execution in `PomegranateQuery::Execute` detects `HasGraph() == false` and routes candidate selection seamlessly to the **SIMD Pulp SQ8 flat scan** path.
- Search queries continue to return correct results at standard SQ8 precision without service disruption.

#include "pomaidb_internal.h"

pdb_status_t pdb_segment_flush(pdb_t* db, const pdb_memtable_t* memtable, const char* out_filepath) {
    if (!db || !memtable || !out_filepath) return PDB_ERR_INVALID_ARGUMENT;
    if (memtable->count == 0) return PDB_SUCCESS;

    pdb_file_t f;
    pdb_status_t st = pdb_file_create(out_filepath, &f);
    if (st != PDB_SUCCESS) return PDB_ERR_IO_FAILURE;

    size_t dim = db->options.dim;
    size_t count = memtable->count;
    size_t vec_bytes = count * dim * sizeof(float);
    size_t id_bytes  = count * sizeof(uint64_t);
    size_t num_words = (count + 63) / 64;
    size_t tomb_bytes = num_words * sizeof(uint64_t);

    pdb_seg_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "POMAISEG", 8);
    hdr.format_version = 1;
    hdr.vector_dim = (uint32_t)dim;
    hdr.metric_type = (uint16_t)db->options.metric;
    hdr.total_vectors = count;
    hdr.segment_lsn = db->current_lsn;

    /* Compute page-aligned offsets */
    uint64_t cur_offset = PDB_PAGE_SIZE; /* Header takes 1 full 4096-byte page */
    hdr.vector_data_offset = cur_offset;
    cur_offset += vec_bytes;

    /* Align next section to 64-byte boundary */
    cur_offset = (cur_offset + 63) & ~63ULL;
    hdr.id_data_offset = cur_offset;
    cur_offset += id_bytes;

    /* Align tombstone section */
    cur_offset = (cur_offset + 63) & ~63ULL;
    hdr.tombstone_offset = cur_offset;
    cur_offset += tomb_bytes;

    hdr.header_crc32c = pdb_crc32c(&hdr, offsetof(pdb_seg_header_t, header_crc32c), 0);

    /* 1. Write Header (4096 Bytes) */
    if (pdb_file_write(&f, &hdr, sizeof(hdr)) != PDB_SUCCESS) {
        pdb_file_close(&f);
        return PDB_ERR_IO_FAILURE;
    }

    /* 2. Write 64-Byte Aligned Vector Payload */
    if (pdb_file_write(&f, memtable->vectors, vec_bytes) != PDB_SUCCESS) {
        pdb_file_close(&f);
        return PDB_ERR_IO_FAILURE;
    }

    /* Padding to ID offset */
    size_t written_so_far = sizeof(hdr) + vec_bytes;
    if (written_so_far < hdr.id_data_offset) {
        size_t pad_len = (size_t)(hdr.id_data_offset - written_so_far);
        char pad[64] = {0};
        if (pdb_file_write(&f, pad, pad_len) != PDB_SUCCESS) {
            pdb_file_close(&f);
            return PDB_ERR_IO_FAILURE;
        }
        written_so_far += pad_len;
    }

    /* 3. Write ID array */
    if (pdb_file_write(&f, memtable->ids, id_bytes) != PDB_SUCCESS) {
        pdb_file_close(&f);
        return PDB_ERR_IO_FAILURE;
    }
    written_so_far += id_bytes;

    /* Padding to tombstone offset */
    if (written_so_far < hdr.tombstone_offset) {
        size_t pad_len = (size_t)(hdr.tombstone_offset - written_so_far);
        char pad[64] = {0};
        if (pdb_file_write(&f, pad, pad_len) != PDB_SUCCESS) {
            pdb_file_close(&f);
            return PDB_ERR_IO_FAILURE;
        }
        written_so_far += pad_len;
    }

    /* 4. Write Tombstone bitset */
    if (pdb_file_write(&f, memtable->tombstones, tomb_bytes) != PDB_SUCCESS) {
        pdb_file_close(&f);
        return PDB_ERR_IO_FAILURE;
    }

    pdb_file_flush(&f);
    pdb_file_close(&f);

    return PDB_SUCCESS;
}

pdb_status_t pdb_segment_open(const char* filepath, pdb_segment_t** out_seg) {
    if (!filepath || !out_seg) return PDB_ERR_INVALID_ARGUMENT;

    pdb_segment_t* seg = (pdb_segment_t*)pa_malloc(sizeof(pdb_segment_t));
    if (!seg) return PDB_ERR_OUT_OF_MEMORY;
    memset(seg, 0, sizeof(pdb_segment_t));
    snprintf(seg->filepath, sizeof(seg->filepath), "%s", filepath);

#if defined(_WIN32) || defined(_WIN64)
    char norm_path[MAX_PATH];
    pdb_normalize_win_path(filepath, norm_path, sizeof(norm_path));
    seg->file_handle = CreateFileA(
        norm_path,
        GENERIC_READ,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        NULL,
        OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL,
        NULL
    );
    if (seg->file_handle == INVALID_HANDLE_VALUE) {
        pa_free(seg);
        return PDB_ERR_IO_FAILURE;
    }

    LARGE_INTEGER file_size;
    if (!GetFileSizeEx(seg->file_handle, &file_size)) {
        CloseHandle(seg->file_handle);
        pa_free(seg);
        return PDB_ERR_IO_FAILURE;
    }
    seg->mmap_size = (size_t)file_size.QuadPart;

    seg->mapping_handle = CreateFileMappingA(
        seg->file_handle,
        NULL,
        PAGE_READONLY,
        0,
        0,
        NULL
    );
    if (!seg->mapping_handle) {
        CloseHandle(seg->file_handle);
        pa_free(seg);
        return PDB_ERR_IO_FAILURE;
    }

    seg->mmap_base = MapViewOfFile(seg->mapping_handle, FILE_MAP_READ, 0, 0, seg->mmap_size);
    if (!seg->mmap_base) {
        CloseHandle(seg->mapping_handle);
        CloseHandle(seg->file_handle);
        pa_free(seg);
        return PDB_ERR_IO_FAILURE;
    }
#else
    seg->fd = open(filepath, O_RDONLY);
    if (seg->fd < 0) {
        pa_free(seg);
        return PDB_ERR_IO_FAILURE;
    }

    struct stat st;
    if (fstat(seg->fd, &st) < 0) {
        close(seg->fd);
        pa_free(seg);
        return PDB_ERR_IO_FAILURE;
    }
    seg->mmap_size = (size_t)st.st_size;

    seg->mmap_base = mmap(NULL, seg->mmap_size, PROT_READ, MAP_SHARED, seg->fd, 0);
    if (seg->mmap_base == MAP_FAILED) {
        close(seg->fd);
        pa_free(seg);
        return PDB_ERR_IO_FAILURE;
    }
#endif

    /* Validate Header */
    seg->header = (const pdb_seg_header_t*)seg->mmap_base;
    if (memcmp(seg->header->magic, "POMAISEG", 8) != 0) {
        pdb_segment_close(seg);
        return PDB_ERR_CORRUPTION;
    }

    uint32_t expected_crc = pdb_crc32c(seg->header, offsetof(pdb_seg_header_t, header_crc32c), 0);
    if (expected_crc != seg->header->header_crc32c) {
        pdb_segment_close(seg);
        return PDB_ERR_CORRUPTION;
    }

    /* Assign zero-copy pointers directly from the OS page cache */
    const char* base = (const char*)seg->mmap_base;
    seg->vectors = (const float*)(base + seg->header->vector_data_offset);
    seg->ids     = (const uint64_t*)(base + seg->header->id_data_offset);
    seg->count   = (size_t)seg->header->total_vectors;
    seg->dim     = (size_t)seg->header->vector_dim;

    /* Copy tombstone bitmap into mutable memory for in-place updates */
    size_t num_words = (seg->count + 63) / 64;
    seg->tombstones = (uint64_t*)pa_malloc(num_words * sizeof(uint64_t));
    if (seg->tombstones) {
        const uint64_t* disk_tomb = (const uint64_t*)(base + seg->header->tombstone_offset);
        memcpy(seg->tombstones, disk_tomb, num_words * sizeof(uint64_t));
    }

    *out_seg = seg;
    return PDB_SUCCESS;
}

void pdb_segment_close(pdb_segment_t* seg) {
    if (!seg) return;
    if (seg->tombstones) {
        pa_free(seg->tombstones);
        seg->tombstones = NULL;
    }
#if defined(_WIN32) || defined(_WIN64)
    if (seg->mmap_base) {
        UnmapViewOfFile(seg->mmap_base);
        seg->mmap_base = NULL;
    }
    if (seg->mapping_handle) {
        CloseHandle(seg->mapping_handle);
        seg->mapping_handle = NULL;
    }
    if (seg->file_handle != INVALID_HANDLE_VALUE) {
        CloseHandle(seg->file_handle);
        seg->file_handle = INVALID_HANDLE_VALUE;
    }
#else
    if (seg->mmap_base && seg->mmap_base != MAP_FAILED) {
        munmap(seg->mmap_base, seg->mmap_size);
        seg->mmap_base = NULL;
    }
    if (seg->fd >= 0) {
        close(seg->fd);
        seg->fd = -1;
    }
#endif
    pa_free(seg);
}

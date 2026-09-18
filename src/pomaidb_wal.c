#include "pomaidb_internal.h"

#include <time.h>

#if defined(_WIN32) || defined(_WIN64)
  #include <direct.h>
  #define pdb_mkdir(path) _mkdir(path)
#else
  #include <sys/types.h>
  #include <sys/stat.h>
  #define pdb_mkdir(path) mkdir(path, 0755)
#endif

pdb_status_t pdb_wal_init(pdb_wal_t* wal, const char* dir_path, bool direct_io) {
    if (!wal) return PDB_ERR_INVALID_ARGUMENT;
    memset(wal, 0, sizeof(pdb_wal_t));
    wal->direct_io = direct_io;
    wal->current_lsn = 0;

    if (!dir_path || dir_path[0] == '\0') {
        return PDB_SUCCESS; /* In-memory execution without WAL */
    }

    pdb_mkdir(dir_path);
    snprintf(wal->filepath, sizeof(wal->filepath), "%s/pomaidb.wal", dir_path);

    /* Open append-binary mode */
    pdb_status_t st = pdb_file_open_append(wal->filepath, &wal->file);
    if (st != PDB_SUCCESS) {
        return PDB_ERR_IO_FAILURE;
    }

    return PDB_SUCCESS;
}

pdb_status_t pdb_wal_append(pdb_wal_t* wal, const pdb_vector_batch_t* batch, uint64_t* out_lsn) {
    if (!wal || !batch) return PDB_ERR_INVALID_ARGUMENT;
    if (!wal->file.is_open) {
        /* In-memory WAL bypass */
        if (out_lsn) *out_lsn = ++wal->current_lsn;
        return PDB_SUCCESS;
    }

    uint64_t lsn = ++wal->current_lsn;
    uint64_t payload_bytes = (batch->count * batch->dim * sizeof(float)) +
                             (batch->count * sizeof(uint64_t));

    pdb_wal_header_t hdr;
    memset(&hdr, 0, sizeof(hdr));
    memcpy(hdr.magic, "POMAIWAL", 8);
    hdr.lsn = lsn;
    hdr.txn_id = lsn;
    hdr.vector_count = (uint32_t)batch->count;
    hdr.vector_dim = (uint32_t)batch->dim;
    hdr.flags = 0x01; /* Commit flag */
    hdr.payload_bytes = payload_bytes;

#if defined(_WIN32) || defined(_WIN64)
    FILETIME ft;
    GetSystemTimeAsFileTime(&ft);
    uint64_t t64 = ((uint64_t)ft.dwHighDateTime << 32) | ft.dwLowDateTime;
    hdr.timestamp_ns = t64 * 100ULL; /* 100-ns intervals to ns */
#else
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    hdr.timestamp_ns = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
#endif

    /* Compute CRC32C over header fields up to header_crc32c */
    hdr.header_crc32c = pdb_crc32c(&hdr, offsetof(pdb_wal_header_t, header_crc32c), 0);

    /* Compute CRC32C over vector payload + IDs */
    uint32_t payload_crc = pdb_crc32c(batch->vectors, batch->count * batch->dim * sizeof(float), 0);
    payload_crc = pdb_crc32c(batch->ids, batch->count * sizeof(uint64_t), payload_crc);

    pdb_wal_footer_t ftr;
    memset(&ftr, 0, sizeof(ftr));
    ftr.magic_tail = 0x21444E455F4C4157ULL; /* "WAL_END!" */
    ftr.lsn_tail = lsn;
    ftr.payload_crc32c = payload_crc;
    ftr.composite_crc32c = pdb_crc32c(&hdr, sizeof(hdr), payload_crc);

    /* Sequential write: Header -> Vectors -> IDs -> Footer */
    if (pdb_file_write(&wal->file, &hdr, sizeof(hdr)) != PDB_SUCCESS) return PDB_ERR_IO_FAILURE;
    if (pdb_file_write(&wal->file, batch->vectors, batch->count * batch->dim * sizeof(float)) != PDB_SUCCESS) return PDB_ERR_IO_FAILURE;
    if (pdb_file_write(&wal->file, batch->ids, batch->count * sizeof(uint64_t)) != PDB_SUCCESS) return PDB_ERR_IO_FAILURE;
    if (pdb_file_write(&wal->file, &ftr, sizeof(ftr)) != PDB_SUCCESS) return PDB_ERR_IO_FAILURE;

    pdb_file_flush(&wal->file);

    if (out_lsn) *out_lsn = lsn;
    return PDB_SUCCESS;
}

pdb_status_t pdb_wal_recover(pdb_wal_t* wal, pdb_t* db) {
    if (!wal || !db) return PDB_ERR_INVALID_ARGUMENT;
    if (!wal->file.is_open) return PDB_SUCCESS;

    pdb_file_close(&wal->file);

    pdb_file_t reader;
    pdb_status_t st = pdb_file_open_read(wal->filepath, &reader);
    if (st != PDB_SUCCESS) {
        pdb_file_open_append(wal->filepath, &wal->file);
        return PDB_SUCCESS;
    }

    pdb_wal_header_t hdr;
    pdb_wal_footer_t ftr;

    while (pdb_file_read_exact(&reader, &hdr, sizeof(hdr)) == PDB_SUCCESS) {
        if (memcmp(hdr.magic, "POMAIWAL", 8) != 0) {
            /* Corruption or end of valid WAL */
            break;
        }

        uint32_t expected_hcrc = pdb_crc32c(&hdr, offsetof(pdb_wal_header_t, header_crc32c), 0);
        if (expected_hcrc != hdr.header_crc32c) {
            /* Torn write in header */
            break;
        }

        size_t vec_bytes = (size_t)hdr.vector_count * (size_t)hdr.vector_dim * sizeof(float);
        size_t id_bytes  = (size_t)hdr.vector_count * sizeof(uint64_t);

        float* vec_buf = (float*)pa_malloc(vec_bytes);
        uint64_t* id_buf = (uint64_t*)pa_malloc(id_bytes);

        if (!vec_buf || !id_buf) {
            if (vec_buf) pa_free(vec_buf);
            if (id_buf) pa_free(id_buf);
            pdb_file_close(&reader);
            pdb_file_open_append(wal->filepath, &wal->file);
            return PDB_ERR_OUT_OF_MEMORY;
        }

        if (pdb_file_read_exact(&reader, vec_buf, vec_bytes) != PDB_SUCCESS ||
            pdb_file_read_exact(&reader, id_buf, id_bytes) != PDB_SUCCESS ||
            pdb_file_read_exact(&reader, &ftr, sizeof(ftr)) != PDB_SUCCESS) {
            /* Torn write at tail; discard partial write */
            pa_free(vec_buf);
            pa_free(id_buf);
            break;
        }

        /* Check footer invariants */
        uint32_t actual_pcrc = pdb_crc32c(vec_buf, vec_bytes, 0);
        actual_pcrc = pdb_crc32c(id_buf, id_bytes, actual_pcrc);

        if (ftr.magic_tail != 0x21444E455F4C4157ULL ||
            ftr.lsn_tail != hdr.lsn ||
            ftr.payload_crc32c != actual_pcrc) {
            /* Inconsistent payload or footer */
            pa_free(vec_buf);
            pa_free(id_buf);
            break;
        }

        /* Replay batch directly into engine memtable */
        pdb_vector_batch_t batch;
        batch.ids = id_buf;
        batch.vectors = vec_buf;
        batch.count = hdr.vector_count;
        batch.dim = hdr.vector_dim;
        batch.metadata_jsons = NULL;

        /* Bypass WAL on replay to avoid circular writes */
        pdb_memtable_t* m = &db->memtable;
        if (m->count + batch.count <= m->capacity) {
            memcpy(m->ids + m->count, batch.ids, batch.count * sizeof(uint64_t));
            memcpy(m->vectors + (m->count * batch.dim), batch.vectors, batch.count * batch.dim * sizeof(float));
            m->count += batch.count;
            db->total_vectors += batch.count;
        }

        if (hdr.lsn > wal->current_lsn) {
            wal->current_lsn = hdr.lsn;
        }

        pa_free(vec_buf);
        pa_free(id_buf);
    }

    pdb_file_close(&reader);
    pdb_file_open_append(wal->filepath, &wal->file);

    db->current_lsn = wal->current_lsn;
    return PDB_SUCCESS;
}

void pdb_wal_close(pdb_wal_t* wal) {
    if (!wal) return;
    pdb_file_close(&wal->file);
}

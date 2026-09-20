use std::ffi::{c_char, c_int, c_void};

pub enum PomaiDbOpaque {}
pub enum PomaiStatusOpaque {}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PomaiOptions {
    pub struct_size: u32,
    pub path: *const c_char,
    pub reserved0: u32,
    pub dim: u32,
    pub search_threads: u32,
    pub fsync_policy: c_int,
    pub memory_budget_bytes: u64,
    pub deadline_ms: u32,

    pub index_type: u8,
    pub hnsw_m: u32,
    pub hnsw_ef_construction: u32,
    pub hnsw_ef_search: u32,
    pub adaptive_threshold: u32,
    pub metric: u8,
    pub edge_profile: u8,
    pub tick_max_ops: u32,
    pub tick_max_ms: u32,
    pub strict_deterministic: bool,

    pub quant_type: u8,
    pub pq_m: u32,
    pub memtable_flush_threshold_mb: u32,
    pub auto_freeze_on_pressure: bool,
    pub max_memtable_mb: u32,
    pub write_coalesce_window_us: u32,
    pub write_coalesce_batch_size: u32,
    pub enable_encryption_at_rest: bool,
    pub encryption_key_hex: *const c_char,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PomaiUpsert {
    pub struct_size: u32,
    pub id: u64,
    pub vector: *const f32,
    pub dim: u32,
    pub metadata: *const u8,
    pub metadata_len: u32,
    pub membrane: *const c_char,
    pub timestamp: u64,
    pub payload: *const u8,
    pub payload_len: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PomaiRecord {
    pub struct_size: u32,
    pub id: u64,
    pub dim: u32,
    pub vector: *const f32,
    pub metadata: *const u8,
    pub metadata_len: u32,
    pub is_deleted: bool,
    pub timestamp: u64,
    pub payload: *const u8,
    pub payload_len: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PomaiQuery {
    pub struct_size: u32,
    pub vector: *const f32,
    pub dim: u32,
    pub topk: u32,
    pub filter_expression: *const c_char,
    pub partition_device_id: *const c_char,
    pub partition_location_id: *const c_char,
    pub deadline_ms: u32,
    pub flags: u32,
    pub membrane: *const c_char,
    pub as_of_ts: u64,
    pub as_of_lsn: u64,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PomaiSearchResults {
    pub struct_size: u32,
    pub count: usize,
    pub ids: *mut u64,
    pub scores: *mut f32,
    pub total_locules_count: u32,
    pub pruned_locules_count: u32,
    pub zero_copy_pointers: *mut c_void,
}

extern "C" {
    pub fn pomai_status_free(status: *mut PomaiStatusOpaque);
    pub fn pomai_status_code(status: *const PomaiStatusOpaque) -> c_int;
    pub fn pomai_status_message(status: *const PomaiStatusOpaque) -> *const c_char;

    pub fn pomai_options_init(opts: *mut PomaiOptions);
    pub fn pomai_open(opts: *const PomaiOptions, out_db: *mut *mut PomaiDbOpaque) -> *mut PomaiStatusOpaque;
    pub fn pomai_close(db: *mut PomaiDbOpaque) -> *mut PomaiStatusOpaque;
    pub fn pomai_flush(db: *mut PomaiDbOpaque) -> *mut PomaiStatusOpaque;
    pub fn pomai_freeze(db: *mut PomaiDbOpaque) -> *mut PomaiStatusOpaque;
    pub fn pomai_freeze_membrane(db: *mut PomaiDbOpaque, membrane: *const c_char) -> *mut PomaiStatusOpaque;
    pub fn pomai_compact(db: *mut PomaiDbOpaque) -> *mut PomaiStatusOpaque;
    pub fn pomai_compact_membrane(db: *mut PomaiDbOpaque, membrane: *const c_char) -> *mut PomaiStatusOpaque;
    pub fn pomai_get_stats_json(db: *mut PomaiDbOpaque, out_json: *mut *mut c_char, out_len: *mut usize) -> *mut PomaiStatusOpaque;

    pub fn pomai_put(db: *mut PomaiDbOpaque, item: *const PomaiUpsert) -> *mut PomaiStatusOpaque;
    pub fn pomai_put_membrane(db: *mut PomaiDbOpaque, membrane: *const c_char, item: *const PomaiUpsert) -> *mut PomaiStatusOpaque;
    pub fn pomai_delete(db: *mut PomaiDbOpaque, id: u64) -> *mut PomaiStatusOpaque;
    pub fn pomai_delete_membrane(db: *mut PomaiDbOpaque, membrane: *const c_char, id: u64) -> *mut PomaiStatusOpaque;
    pub fn pomai_exists(db: *mut PomaiDbOpaque, id: u64, out_exists: *mut bool) -> *mut PomaiStatusOpaque;
    pub fn pomai_exists_membrane(db: *mut PomaiDbOpaque, membrane: *const c_char, id: u64, out_exists: *mut bool) -> *mut PomaiStatusOpaque;
    pub fn pomai_get(db: *mut PomaiDbOpaque, id: u64, out_record: *mut *mut PomaiRecord) -> *mut PomaiStatusOpaque;
    pub fn pomai_get_membrane(db: *mut PomaiDbOpaque, membrane: *const c_char, id: u64, out_record: *mut *mut PomaiRecord) -> *mut PomaiStatusOpaque;
    pub fn pomai_record_free(record: *mut PomaiRecord);

    pub fn pomai_search(db: *mut PomaiDbOpaque, query: *const PomaiQuery, out: *mut *mut PomaiSearchResults) -> *mut PomaiStatusOpaque;
    pub fn pomai_search_membrane(db: *mut PomaiDbOpaque, membrane: *const c_char, query: *const PomaiQuery, out: *mut *mut PomaiSearchResults) -> *mut PomaiStatusOpaque;
    pub fn pomai_search_results_free(results: *mut PomaiSearchResults);

    pub fn pomai_create_membrane_kind(db: *mut PomaiDbOpaque, name: *const c_char, dim: u32, kind: u32) -> *mut PomaiStatusOpaque;
    pub fn pomai_drop_membrane(db: *mut PomaiDbOpaque, name: *const c_char) -> *mut PomaiStatusOpaque;
    pub fn pomai_open_membrane(db: *mut PomaiDbOpaque, name: *const c_char) -> *mut PomaiStatusOpaque;
    pub fn pomai_close_membrane(db: *mut PomaiDbOpaque, name: *const c_char) -> *mut PomaiStatusOpaque;
    pub fn pomai_list_membranes_json(db: *mut PomaiDbOpaque, out_json: *mut *mut c_char, out_len: *mut usize) -> *mut PomaiStatusOpaque;

    pub fn pomai_free(ptr: *mut c_void);
}
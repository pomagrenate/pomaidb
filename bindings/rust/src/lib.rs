pub mod sys;

use std::ffi::{c_void, CStr, CString};
use std::mem::MaybeUninit;
use std::ptr;

#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("PomaiDB error (code {code}): {message}")]
    Native { code: i32, message: String },
    #[error("Invalid argument: {0}")]
    InvalidArgument(String),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Nul byte in C string: {0}")]
    Nul(#[from] std::ffi::NulError),
}

pub type Result<T> = std::result::Result<T, Error>;

fn check_status(st: *mut sys::PomaiStatusOpaque) -> Result<()> {
    if st.is_null() {
        return Ok(());
    }
    unsafe {
        let code = sys::pomai_status_code(st);
        let msg_ptr = sys::pomai_status_message(st);
        let message = if msg_ptr.is_null() {
            String::new()
        } else {
            CStr::from_ptr(msg_ptr).to_string_lossy().into_owned()
        };
        sys::pomai_status_free(st);
        Err(Error::Native { code, message })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricType {
    L2 = 0,
    InnerProduct = 1,
    Cosine = 2,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantType {
    None = 0,
    SQ8 = 1,
    FP16 = 2,
    Bit = 3,
    PQ8 = 4,
}

#[derive(Debug, Clone)]
pub struct Options {
    pub path: String,
    pub dim: usize,
    pub shards: usize,
    pub metric: MetricType,
    pub quant_type: QuantType,
    pub memory_budget_bytes: u64,
}

impl Options {
    pub fn new(path: impl Into<String>, dim: usize) -> Self {
        Self {
            path: path.into(),
            dim,
            shards: 1,
            metric: MetricType::L2,
            quant_type: QuantType::None,
            memory_budget_bytes: 0,
        }
    }

    pub fn shards(mut self, shards: usize) -> Self {
        self.shards = shards;
        self
    }

    pub fn metric(mut self, metric: MetricType) -> Self {
        self.metric = metric;
        self
    }

    pub fn quant_type(mut self, quant_type: QuantType) -> Self {
        self.quant_type = quant_type;
        self
    }

    pub fn memory_budget_bytes(mut self, bytes: u64) -> Self {
        self.memory_budget_bytes = bytes;
        self
    }
}

#[derive(Debug, Clone)]
pub struct Record {
    pub id: u64,
    pub vector: Vec<f32>,
    pub dim: usize,
    pub timestamp: u64,
    pub payload: Option<Vec<u8>>,
    pub is_deleted: bool,
}

#[derive(Debug, Clone)]
pub struct SearchResult {
    pub id: u64,
    pub score: f32,
}

#[derive(Debug, Clone, Default)]
pub struct PutOptions<'a> {
    pub membrane: Option<&'a str>,
    pub timestamp: u64,
    pub payload: Option<&'a [u8]>,
}

#[derive(Debug, Clone, Default)]
pub struct SearchOptions<'a> {
    pub membrane: Option<&'a str>,
    pub filter_expression: Option<&'a str>,
    pub as_of_ts: u64,
    pub as_of_lsn: u64,
}

pub struct Database {
    handle: *mut sys::PomaiDbOpaque,
}

unsafe impl Send for Database {}
unsafe impl Sync for Database {}

impl Database {
    pub fn open(options: &Options) -> Result<Self> {
        let path_c = CString::new(options.path.as_str())?;
        unsafe {
            let mut opts = MaybeUninit::<sys::PomaiOptions>::zeroed().assume_init();
            opts.struct_size = std::mem::size_of::<sys::PomaiOptions>() as u32;
            sys::pomai_options_init(&mut opts);
            opts.struct_size = std::mem::size_of::<sys::PomaiOptions>() as u32;
            opts.path = path_c.as_ptr();
            opts.dim = options.dim as u32;
            opts.shards = options.shards as u32;
            opts.metric = options.metric as u8;
            opts.quant_type = options.quant_type as u8;
            if options.memory_budget_bytes > 0 {
                opts.memory_budget_bytes = options.memory_budget_bytes;
            }

            let mut out_db: *mut sys::PomaiDbOpaque = ptr::null_mut();
            check_status(sys::pomai_open(&opts, &mut out_db))?;
            Ok(Self { handle: out_db })
        }
    }

    pub fn put(&self, id: u64, vector: &[f32]) -> Result<()> {
        self.put_with_options(id, vector, &PutOptions::default())
    }

    pub fn put_with_options(&self, id: u64, vector: &[f32], options: &PutOptions) -> Result<()> {
        let membrane_c = match options.membrane {
            Some(m) => Some(CString::new(m)?),
            None => None,
        };

        let upsert = sys::PomaiUpsert {
            struct_size: std::mem::size_of::<sys::PomaiUpsert>() as u32,
            id,
            vector: vector.as_ptr(),
            dim: vector.len() as u32,
            metadata: ptr::null(),
            metadata_len: 0,
            membrane: membrane_c.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
            timestamp: options.timestamp,
            payload: options.payload.map_or(ptr::null(), |p| p.as_ptr()),
            payload_len: options.payload.map_or(0, |p| p.len() as u32),
        };

        unsafe {
            if let Some(ref m) = membrane_c {
                check_status(sys::pomai_put_membrane(self.handle, m.as_ptr(), &upsert))
            } else {
                check_status(sys::pomai_put(self.handle, &upsert))
            }
        }
    }

    pub fn get(&self, id: u64) -> Result<Option<Record>> {
        self.get_membrane(None, id)
    }

    pub fn get_membrane(&self, membrane: Option<&str>, id: u64) -> Result<Option<Record>> {
        let membrane_c = match membrane {
            Some(m) => Some(CString::new(m)?),
            None => None,
        };

        unsafe {
            let mut out_rec: *mut sys::PomaiRecord = ptr::null_mut();
            let st = if let Some(ref m) = membrane_c {
                sys::pomai_get_membrane(self.handle, m.as_ptr(), id, &mut out_rec)
            } else {
                sys::pomai_get(self.handle, id, &mut out_rec)
            };

            if let Err(e) = check_status(st) {
                match e {
                    Error::Native { code: 2, .. } => return Ok(None), // Not found
                    other => return Err(other),
                }
            }

            if out_rec.is_null() {
                return Ok(None);
            }

            let rec_ref = &*out_rec;
            let vec_slice = std::slice::from_raw_parts(rec_ref.vector, rec_ref.dim as usize);
            let payload = if !rec_ref.payload.is_null() && rec_ref.payload_len > 0 {
                Some(std::slice::from_raw_parts(rec_ref.payload, rec_ref.payload_len as usize).to_vec())
            } else {
                None
            };

            let record = Record {
                id: rec_ref.id,
                vector: vec_slice.to_vec(),
                dim: rec_ref.dim as usize,
                timestamp: rec_ref.timestamp,
                payload,
                is_deleted: rec_ref.is_deleted,
            };

            sys::pomai_record_free(out_rec);
            Ok(Some(record))
        }
    }

    pub fn exists(&self, id: u64) -> Result<bool> {
        self.exists_membrane(None, id)
    }

    pub fn exists_membrane(&self, membrane: Option<&str>, id: u64) -> Result<bool> {
        let membrane_c = match membrane {
            Some(m) => Some(CString::new(m)?),
            None => None,
        };

        unsafe {
            let mut out_exists = false;
            if let Some(ref m) = membrane_c {
                check_status(sys::pomai_exists_membrane(self.handle, m.as_ptr(), id, &mut out_exists))?;
            } else {
                check_status(sys::pomai_exists(self.handle, id, &mut out_exists))?;
            }
            Ok(out_exists)
        }
    }

    pub fn delete(&self, id: u64) -> Result<()> {
        self.delete_membrane(None, id)
    }

    pub fn delete_membrane(&self, membrane: Option<&str>, id: u64) -> Result<()> {
        let membrane_c = match membrane {
            Some(m) => Some(CString::new(m)?),
            None => None,
        };

        unsafe {
            if let Some(ref m) = membrane_c {
                check_status(sys::pomai_delete_membrane(self.handle, m.as_ptr(), id))
            } else {
                check_status(sys::pomai_delete(self.handle, id))
            }
        }
    }

    pub fn search(&self, query_vector: &[f32], topk: usize) -> Result<Vec<SearchResult>> {
        self.search_with_options(query_vector, topk, &SearchOptions::default())
    }

    pub fn search_with_options(&self, query_vector: &[f32], topk: usize, options: &SearchOptions) -> Result<Vec<SearchResult>> {
        let membrane_c = match options.membrane {
            Some(m) => Some(CString::new(m)?),
            None => None,
        };
        let filter_c = match options.filter_expression {
            Some(f) => Some(CString::new(f)?),
            None => None,
        };

        let q = sys::PomaiQuery {
            struct_size: std::mem::size_of::<sys::PomaiQuery>() as u32,
            vector: query_vector.as_ptr(),
            dim: query_vector.len() as u32,
            topk: topk as u32,
            filter_expression: filter_c.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
            partition_device_id: ptr::null(),
            partition_location_id: ptr::null(),
            deadline_ms: 0,
            flags: 0,
            membrane: membrane_c.as_ref().map_or(ptr::null(), |c| c.as_ptr()),
            as_of_ts: options.as_of_ts,
            as_of_lsn: options.as_of_lsn,
        };

        unsafe {
            let mut out_res: *mut sys::PomaiSearchResults = ptr::null_mut();
            if let Some(ref m) = membrane_c {
                check_status(sys::pomai_search_membrane(self.handle, m.as_ptr(), &q, &mut out_res))?;
            } else {
                check_status(sys::pomai_search(self.handle, &q, &mut out_res))?;
            }

            if out_res.is_null() {
                return Ok(Vec::new());
            }

            let res_ref = &*out_res;
            let mut hits = Vec::with_capacity(res_ref.count);
            if res_ref.count > 0 && !res_ref.ids.is_null() && !res_ref.scores.is_null() {
                let ids = std::slice::from_raw_parts(res_ref.ids, res_ref.count);
                let scores = std::slice::from_raw_parts(res_ref.scores, res_ref.count);
                for i in 0..res_ref.count {
                    hits.push(SearchResult {
                        id: ids[i],
                        score: scores[i],
                    });
                }
            }

            sys::pomai_search_results_free(out_res);
            Ok(hits)
        }
    }

    pub fn flush(&self) -> Result<()> {
        unsafe { check_status(sys::pomai_flush(self.handle)) }
    }

    pub fn freeze(&self, membrane: Option<&str>) -> Result<()> {
        let membrane_c = match membrane {
            Some(m) => Some(CString::new(m)?),
            None => None,
        };
        unsafe {
            if let Some(ref m) = membrane_c {
                check_status(sys::pomai_freeze_membrane(self.handle, m.as_ptr()))
            } else {
                check_status(sys::pomai_freeze(self.handle))
            }
        }
    }

    pub fn compact(&self, membrane: Option<&str>) -> Result<()> {
        let membrane_c = match membrane {
            Some(m) => Some(CString::new(m)?),
            None => None,
        };
        unsafe {
            if let Some(ref m) = membrane_c {
                check_status(sys::pomai_compact_membrane(self.handle, m.as_ptr()))
            } else {
                check_status(sys::pomai_compact(self.handle))
            }
        }
    }

    pub fn create_membrane(&self, name: &str, dim: usize, shard_count: usize) -> Result<()> {
        let name_c = CString::new(name)?;
        unsafe {
            check_status(sys::pomai_create_membrane_kind(
                self.handle,
                name_c.as_ptr(),
                dim as u32,
                shard_count as u32,
                0,
            ))
        }
    }

    pub fn drop_membrane(&self, name: &str) -> Result<()> {
        let name_c = CString::new(name)?;
        unsafe { check_status(sys::pomai_drop_membrane(self.handle, name_c.as_ptr())) }
    }

    pub fn open_membrane(&self, name: &str) -> Result<()> {
        let name_c = CString::new(name)?;
        unsafe { check_status(sys::pomai_open_membrane(self.handle, name_c.as_ptr())) }
    }

    pub fn close_membrane(&self, name: &str) -> Result<()> {
        let name_c = CString::new(name)?;
        unsafe { check_status(sys::pomai_close_membrane(self.handle, name_c.as_ptr())) }
    }

    pub fn list_membranes(&self) -> Result<Vec<String>> {
        unsafe {
            let mut out_json: *mut std::ffi::c_char = ptr::null_mut();
            let mut out_len: usize = 0;
            check_status(sys::pomai_list_membranes_json(self.handle, &mut out_json, &mut out_len))?;

            if out_json.is_null() {
                return Ok(Vec::new());
            }

            let slice = std::slice::from_raw_parts(out_json as *const u8, out_len);
            let s = std::str::from_utf8(slice).map_err(|e| Error::InvalidArgument(e.to_string()))?;
            let list: Vec<String> = serde_json::from_str(s)?;
            sys::pomai_free(out_json as *mut c_void);
            Ok(list)
        }
    }

    pub fn get_stats(&self) -> Result<serde_json::Value> {
        unsafe {
            let mut out_json: *mut std::ffi::c_char = ptr::null_mut();
            let mut out_len: usize = 0;
            check_status(sys::pomai_get_stats_json(self.handle, &mut out_json, &mut out_len))?;

            if out_json.is_null() {
                return Ok(serde_json::Value::Null);
            }

            let slice = std::slice::from_raw_parts(out_json as *const u8, out_len);
            let s = std::str::from_utf8(slice).map_err(|e| Error::InvalidArgument(e.to_string()))?;
            let v: serde_json::Value = serde_json::from_str(s)?;
            sys::pomai_free(out_json as *mut c_void);
            Ok(v)
        }
    }

    pub fn close(&mut self) -> Result<()> {
        if !self.handle.is_null() {
            unsafe {
                let st = sys::pomai_close(self.handle);
                self.handle = ptr::null_mut();
                check_status(st)?;
            }
        }
        Ok(())
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        let _ = self.close();
    }
}
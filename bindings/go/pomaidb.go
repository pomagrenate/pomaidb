package pomaidb

/*
#cgo CFLAGS: -I${SRCDIR}/include/pomai -I${SRCDIR}/../../include/pomai
#cgo LDFLAGS: -L${SRCDIR}/lib -L${SRCDIR}/../../build -lpomai_c
#include "c_api.h"
#include <stdlib.h>
*/
import "C"
import (
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"sync"
	"unsafe"
)

type MetricType uint8

const (
	MetricL2           MetricType = 0
	MetricInnerProduct MetricType = 1
	MetricCosine       MetricType = 2
)

type QuantType uint8

const (
	QuantNone QuantType = 0
	QuantSQ8  QuantType = 1
	QuantFP16 QuantType = 2
	QuantBit  QuantType = 3
	QuantPQ8  QuantType = 4
)

type Options struct {
	Path              string
	Dim               uint32
	Shards            uint32
	Metric            MetricType
	QuantType         QuantType
	MemoryBudgetBytes uint64
}

type Record struct {
	ID        uint64    `json:"id"`
	Vector    []float32 `json:"vector"`
	Dim       uint32    `json:"dim"`
	Timestamp uint64    `json:"timestamp"`
	Payload   []byte    `json:"payload,omitempty"`
	IsDeleted bool      `json:"is_deleted"`
}

type SearchResult struct {
	ID    uint64  `json:"id"`
	Score float32 `json:"score"`
}

type PutOptions struct {
	Membrane  string
	Timestamp uint64
	Payload   []byte
}

type SearchOptions struct {
	Membrane         string
	FilterExpression string
	AsOfTs           uint64
	AsOfLsn          uint64
}

type DB struct {
	mu     sync.RWMutex
	handle *C.pomai_db_t
}

func checkStatus(st *C.pomai_status_t) error {
	if st == nil {
		return nil
	}
	code := int(C.pomai_status_code(st))
	cMsg := C.pomai_status_message(st)
	msg := ""
	if cMsg != nil {
		msg = C.GoString(cMsg)
	}
	C.pomai_status_free(st)
	return fmt.Errorf("pomaidb error (code %d): %s", code, msg)
}

func Open(opts Options) (*DB, error) {
	if opts.Path == "" {
		return nil, errors.New("database path required")
	}

	var cOpts C.pomai_options_t
	cOpts.struct_size = C.uint32_t(unsafe.Sizeof(cOpts))
	C.pomai_options_init(&cOpts)
	cOpts.struct_size = C.uint32_t(unsafe.Sizeof(cOpts))

	cPath := C.CString(opts.Path)
	defer C.free(unsafe.Pointer(cPath))

	cOpts.path = cPath
	cOpts.dim = C.uint32_t(opts.Dim)
	if opts.Shards > 0 {
		cOpts.shards = C.uint32_t(opts.Shards)
	} else {
		cOpts.shards = 1
	}
	cOpts.metric = C.uint8_t(opts.Metric)
	cOpts.quant_type = C.uint8_t(opts.QuantType)
	if opts.MemoryBudgetBytes > 0 {
		cOpts.memory_budget_bytes = C.uint64_t(opts.MemoryBudgetBytes)
	}

	var outDb *C.pomai_db_t
	if err := checkStatus(C.pomai_open(&cOpts, &outDb)); err != nil {
		return nil, err
	}

	db := &DB{handle: outDb}
	runtime.SetFinalizer(db, func(d *DB) {
		_ = d.Close()
	})
	return db, nil
}

func (db *DB) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	if db.handle == nil {
		return nil
	}
	h := db.handle
	db.handle = nil
	return checkStatus(C.pomai_close(h))
}

func (db *DB) Put(id uint64, vector []float32) error {
	return db.PutWithOptions(id, vector, PutOptions{})
}

func (db *DB) PutWithOptions(id uint64, vector []float32, options PutOptions) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.handle == nil {
		return errors.New("database closed")
	}
	if len(vector) == 0 {
		return errors.New("empty vector")
	}

	var pinner runtime.Pinner
	pinner.Pin(&vector[0])
	if len(options.Payload) > 0 {
		pinner.Pin(&options.Payload[0])
	}
	defer pinner.Unpin()

	var item C.pomai_upsert_t
	item.struct_size = C.uint32_t(unsafe.Sizeof(item))
	item.id = C.uint64_t(id)
	item.vector = (*C.float)(unsafe.Pointer(&vector[0]))
	item.dim = C.uint32_t(len(vector))
	item.timestamp = C.uint64_t(options.Timestamp)

	if len(options.Payload) > 0 {
		item.payload = (*C.uint8_t)(unsafe.Pointer(&options.Payload[0]))
		item.payload_len = C.uint32_t(len(options.Payload))
	}

	if options.Membrane != "" {
		cMemb := C.CString(options.Membrane)
		defer C.free(unsafe.Pointer(cMemb))
		item.membrane = cMemb
		return checkStatus(C.pomai_put_membrane(db.handle, cMemb, &item))
	}

	return checkStatus(C.pomai_put(db.handle, &item))
}

func (db *DB) Get(id uint64) (*Record, error) {
	return db.GetMembrane("", id)
}

func (db *DB) GetMembrane(membrane string, id uint64) (*Record, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.handle == nil {
		return nil, errors.New("database closed")
	}

	var outRec *C.pomai_record_t
	var st *C.pomai_status_t

	if membrane != "" {
		cMemb := C.CString(membrane)
		defer C.free(unsafe.Pointer(cMemb))
		st = C.pomai_get_membrane(db.handle, cMemb, C.uint64_t(id), &outRec)
	} else {
		st = C.pomai_get(db.handle, C.uint64_t(id), &outRec)
	}

	if err := checkStatus(st); err != nil {
		return nil, err
	}
	if outRec == nil {
		return nil, nil
	}
	defer C.pomai_record_free(outRec)

	dim := int(outRec.dim)
	vec := make([]float32, dim)
	if dim > 0 && outRec.vector != nil {
		copy(vec, unsafe.Slice((*float32)(unsafe.Pointer(outRec.vector)), dim))
	}

	var payload []byte
	payloadLen := int(outRec.payload_len)
	if payloadLen > 0 && outRec.payload != nil {
		payload = make([]byte, payloadLen)
		copy(payload, unsafe.Slice((*byte)(unsafe.Pointer(outRec.payload)), payloadLen))
	}

	return &Record{
		ID:        uint64(outRec.id),
		Vector:    vec,
		Dim:       uint32(dim),
		Timestamp: uint64(outRec.timestamp),
		Payload:   payload,
		IsDeleted: bool(outRec.is_deleted),
	}, nil
}

func (db *DB) Exists(id uint64) (bool, error) {
	return db.ExistsMembrane("", id)
}

func (db *DB) ExistsMembrane(membrane string, id uint64) (bool, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.handle == nil {
		return false, errors.New("database closed")
	}

	var outExists C.bool
	var st *C.pomai_status_t

	if membrane != "" {
		cMemb := C.CString(membrane)
		defer C.free(unsafe.Pointer(cMemb))
		st = C.pomai_exists_membrane(db.handle, cMemb, C.uint64_t(id), &outExists)
	} else {
		st = C.pomai_exists(db.handle, C.uint64_t(id), &outExists)
	}

	if err := checkStatus(st); err != nil {
		return false, err
	}
	return bool(outExists), nil
}

func (db *DB) Delete(id uint64) error {
	return db.DeleteMembrane("", id)
}

func (db *DB) DeleteMembrane(membrane string, id uint64) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.handle == nil {
		return errors.New("database closed")
	}

	if membrane != "" {
		cMemb := C.CString(membrane)
		defer C.free(unsafe.Pointer(cMemb))
		return checkStatus(C.pomai_delete_membrane(db.handle, cMemb, C.uint64_t(id)))
	}
	return checkStatus(C.pomai_delete(db.handle, C.uint64_t(id)))
}

func (db *DB) Search(queryVector []float32, topK int) ([]SearchResult, error) {
	return db.SearchWithOptions(queryVector, topK, SearchOptions{})
}

func (db *DB) SearchWithOptions(queryVector []float32, topK int, options SearchOptions) ([]SearchResult, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.handle == nil {
		return nil, errors.New("database closed")
	}
	if len(queryVector) == 0 {
		return nil, errors.New("empty query vector")
	}

	var pinner runtime.Pinner
	pinner.Pin(&queryVector[0])
	defer pinner.Unpin()

	var q C.pomai_query_t
	q.struct_size = C.uint32_t(unsafe.Sizeof(q))
	q.vector = (*C.float)(unsafe.Pointer(&queryVector[0]))
	q.dim = C.uint32_t(len(queryVector))
	q.topk = C.uint32_t(topK)
	q.as_of_ts = C.uint64_t(options.AsOfTs)
	q.as_of_lsn = C.uint64_t(options.AsOfLsn)

	var cMemb *C.char
	if options.Membrane != "" {
		cMemb = C.CString(options.Membrane)
		defer C.free(unsafe.Pointer(cMemb))
		q.membrane = cMemb
	}

	var cFilter *C.char
	if options.FilterExpression != "" {
		cFilter = C.CString(options.FilterExpression)
		defer C.free(unsafe.Pointer(cFilter))
		q.filter_expression = cFilter
	}

	var outRes *C.pomai_search_results_t
	var st *C.pomai_status_t

	if cMemb != nil {
		st = C.pomai_search_membrane(db.handle, cMemb, &q, &outRes)
	} else {
		st = C.pomai_search(db.handle, &q, &outRes)
	}

	if err := checkStatus(st); err != nil {
		return nil, err
	}
	if outRes == nil {
		return []SearchResult{}, nil
	}
	defer C.pomai_search_results_free(outRes)

	count := int(outRes.count)
	results := make([]SearchResult, count)
	if count > 0 && outRes.ids != nil && outRes.scores != nil {
		ids := unsafe.Slice((*uint64)(unsafe.Pointer(outRes.ids)), count)
		scores := unsafe.Slice((*float32)(unsafe.Pointer(outRes.scores)), count)
		for i := 0; i < count; i++ {
			results[i] = SearchResult{
				ID:    ids[i],
				Score: scores[i],
			}
		}
	}
	return results, nil
}

func (db *DB) Flush() error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.handle == nil {
		return errors.New("database closed")
	}
	return checkStatus(C.pomai_flush(db.handle))
}

func (db *DB) Freeze(membrane ...string) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.handle == nil {
		return errors.New("database closed")
	}

	if len(membrane) > 0 && membrane[0] != "" {
		cMemb := C.CString(membrane[0])
		defer C.free(unsafe.Pointer(cMemb))
		return checkStatus(C.pomai_freeze_membrane(db.handle, cMemb))
	}
	return checkStatus(C.pomai_freeze(db.handle))
}

func (db *DB) Compact(membrane ...string) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.handle == nil {
		return errors.New("database closed")
	}

	if len(membrane) > 0 && membrane[0] != "" {
		cMemb := C.CString(membrane[0])
		defer C.free(unsafe.Pointer(cMemb))
		return checkStatus(C.pomai_compact_membrane(db.handle, cMemb))
	}
	return checkStatus(C.pomai_compact(db.handle))
}

func (db *DB) CreateMembrane(name string, dim uint32, shardCount uint32) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.handle == nil {
		return errors.New("database closed")
	}

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	shards := shardCount
	if shards == 0 {
		shards = 1
	}
	return checkStatus(C.pomai_create_membrane_kind(db.handle, cName, C.uint32_t(dim), C.uint32_t(shards), 0))
}

func (db *DB) DropMembrane(name string) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.handle == nil {
		return errors.New("database closed")
	}

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.pomai_drop_membrane(db.handle, cName))
}

func (db *DB) OpenMembrane(name string) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.handle == nil {
		return errors.New("database closed")
	}

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.pomai_open_membrane(db.handle, cName))
}

func (db *DB) CloseMembrane(name string) error {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.handle == nil {
		return errors.New("database closed")
	}

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	return checkStatus(C.pomai_close_membrane(db.handle, cName))
}

func (db *DB) ListMembranes() ([]string, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.handle == nil {
		return nil, errors.New("database closed")
	}

	var outJson *C.char
	var outLen C.size_t
	if err := checkStatus(C.pomai_list_membranes_json(db.handle, &outJson, &outLen)); err != nil {
		return nil, err
	}
	if outJson == nil {
		return []string{}, nil
	}
	defer C.pomai_free(unsafe.Pointer(outJson))

	data := C.GoBytes(unsafe.Pointer(outJson), C.int(outLen))
	var membranes []string
	if err := json.Unmarshal(data, &membranes); err != nil {
		return nil, err
	}
	return membranes, nil
}

func (db *DB) GetStats() (map[string]interface{}, error) {
	db.mu.RLock()
	defer db.mu.RUnlock()

	if db.handle == nil {
		return nil, errors.New("database closed")
	}

	var outJson *C.char
	var outLen C.size_t
	if err := checkStatus(C.pomai_get_stats_json(db.handle, &outJson, &outLen)); err != nil {
		return nil, err
	}
	if outJson == nil {
		return map[string]interface{}{}, nil
	}
	defer C.pomai_free(unsafe.Pointer(outJson))

	data := C.GoBytes(unsafe.Pointer(outJson), C.int(outLen))
	var stats map[string]interface{}
	if err := json.Unmarshal(data, &stats); err != nil {
		return nil, err
	}
	return stats, nil
}
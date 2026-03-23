#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <cstdio>

struct cheese_file;
struct cheese_mmap;
struct cheese_mlock;

using cheese_files  = std::vector<std::unique_ptr<cheese_file>>;
using cheese_mmaps  = std::vector<std::unique_ptr<cheese_mmap>>;
using cheese_mlocks = std::vector<std::unique_ptr<cheese_mlock>>;

struct cheese_file {
    cheese_file(const char * fname, const char * mode, bool use_direct_io = false);
    ~cheese_file();

    size_t tell() const;
    size_t size() const;

    int file_id() const; // fileno overload

    void seek(size_t offset, int whence) const;

    void read_raw(void * ptr, size_t len);
    void read_raw_unsafe(void * ptr, size_t len);
    void read_aligned_chunk(void * dest, size_t size);
    uint32_t read_u32();

    void write_raw(const void * ptr, size_t len) const;
    void write_u32(uint32_t val) const;

    size_t read_alignment() const;
    bool has_direct_io() const;
private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct cheese_mmap {
    cheese_mmap(const cheese_mmap &) = delete;
    cheese_mmap(struct cheese_file * file, size_t prefetch = (size_t) -1, bool numa = false);
    ~cheese_mmap();

    size_t size() const;
    void * addr() const;

    void unmap_fragment(size_t first, size_t last);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

struct cheese_mlock {
    cheese_mlock();
    ~cheese_mlock();

    void init(void * ptr);
    void grow_to(size_t target_size);

    static const bool SUPPORTED;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

size_t cheese_path_max();

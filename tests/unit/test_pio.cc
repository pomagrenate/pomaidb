// test/test_pio.cc — Unit and verification test suite for pio
// Copyright 2026 PomaiDB / pio authors. MIT License.

#include "pio/pio.h"
#include <iostream>
#include <cassert>
#include <string>
#include <vector>
#include <cstring>
#include <chrono>

#define TEST_ASSERT(cond) \
    do { \
        if (!(cond)) { \
            std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ << " -> " << #cond << std::endl; \
            std::exit(1); \
        } \
    } while(0)

#define TEST_ASSERT_OK(status_expr) \
    do { \
        pio::Status s = (status_expr); \
        if (!s.ok()) { \
            std::cerr << "Expected OK at " << __FILE__ << ":" << __LINE__ \
                      << " but got: " << s.ToString() << std::endl; \
            std::exit(1); \
        } \
    } while(0)

void TestSlice() {
    std::cout << "[TEST] TestSlice..." << std::flush;
    pio::Slice empty;
    TEST_ASSERT(empty.empty());
    TEST_ASSERT(empty.size() == 0);

    std::string text = "Hello PomaiDB Direct I/O";
    pio::Slice s1(text);
    TEST_ASSERT(!s1.empty());
    TEST_ASSERT(s1.size() == text.size());
    TEST_ASSERT(s1.ToString() == text);
    TEST_ASSERT(s1.starts_with(pio::Slice("Hello")));
    TEST_ASSERT(!s1.starts_with(pio::Slice("World")));

    pio::Slice s2(text.data(), 5);
    TEST_ASSERT(s2.ToString() == "Hello");
    TEST_ASSERT(s1 != s2);
    TEST_ASSERT(s2 == pio::Slice("Hello"));

    s2.remove_prefix(2);
    TEST_ASSERT(s2.ToString() == "llo");
    std::cout << " PASSED" << std::endl;
}

void TestStatus() {
    std::cout << "[TEST] TestStatus..." << std::flush;
    pio::Status ok = pio::Status::Ok();
    TEST_ASSERT(ok.ok());
    TEST_ASSERT(ok.ToString() == "OK");

    pio::Status nf = pio::Status::NotFound("missing_file.bin");
    TEST_ASSERT(!nf.ok());
    TEST_ASSERT(nf.IsNotFound());
    TEST_ASSERT(nf.ToString().find("NotFound") != std::string::npos);

    pio::Status io = pio::Status::IOError("disk full");
    TEST_ASSERT(!io.ok());
    TEST_ASSERT(io.IsIOError());

    pio::Status corrupt = pio::Status::Corruption("crc mismatch");
    TEST_ASSERT(corrupt.IsCorruption());

    pio::Status inv = pio::Status::InvalidArgument("bad size");
    TEST_ASSERT(inv.IsInvalidArgument());

    pio::Status copied = nf;
    TEST_ASSERT(copied.IsNotFound());
    pio::Status moved = std::move(copied);
    TEST_ASSERT(moved.IsNotFound());
    std::cout << " PASSED" << std::endl;
}

void TestDirectBuffer() {
    std::cout << "[TEST] TestDirectBuffer..." << std::flush;
    pio::AlignedBuffer buf(8192);
    TEST_ASSERT(buf.capacity() >= 8192);
    TEST_ASSERT(buf.size() == 0);
    TEST_ASSERT(buf.IsAligned());
    TEST_ASSERT(reinterpret_cast<uintptr_t>(buf.data()) % pio::kSectorSize == 0);

    const char* sample = "Zero-copy Direct I/O Aligned Sector Buffer";
    size_t sample_len = std::strlen(sample);
    std::memcpy(buf.data(), sample, sample_len);
    buf.set_size(sample_len);
    TEST_ASSERT(buf.size() == sample_len);
    TEST_ASSERT(std::memcmp(buf.data(), sample, sample_len) == 0);

    // Test resize
    buf.resize(16384);
    TEST_ASSERT(buf.capacity() >= 16384);
    TEST_ASSERT(buf.IsAligned());
    TEST_ASSERT(std::memcmp(buf.data(), sample, sample_len) == 0);

    // Test move
    pio::AlignedBuffer moved(std::move(buf));
    TEST_ASSERT(moved.capacity() >= 16384);
    TEST_ASSERT(moved.size() == sample_len);
    TEST_ASSERT(moved.IsAligned());
    TEST_ASSERT(buf.data() == nullptr);
    TEST_ASSERT(buf.capacity() == 0);

    std::cout << " PASSED" << std::endl;
}

void TestWritableAndSequentialFile() {
    std::cout << "[TEST] TestWritableAndSequentialFile..." << std::flush;
    std::string path = "pio_test_seq.bin";
    (void)pio::FileSystem::DeleteFile(path);

    std::unique_ptr<pio::WritableFile> wfile;
    TEST_ASSERT_OK(pio::FileSystem::NewWritableFile(path, &wfile));

    std::string chunk1 = "SegmentHeader_Magic_0xDEADBEEF\n";
    std::string chunk2 = "LoculePayloadBlock_1234567890\n";
    std::string large_chunk(128 * 1024, 'X'); // 128KB to trigger direct unbuffered write

    TEST_ASSERT_OK(wfile->Append(pio::Slice(chunk1)));
    TEST_ASSERT_OK(wfile->Append(pio::Slice(chunk2)));
    TEST_ASSERT_OK(wfile->Append(pio::Slice(large_chunk)));
    TEST_ASSERT_OK(wfile->Flush());
    TEST_ASSERT_OK(wfile->Sync());

    uint64_t expected_size = chunk1.size() + chunk2.size() + large_chunk.size();
    TEST_ASSERT(wfile->BytesWritten() == expected_size);
    TEST_ASSERT_OK(wfile->Close());

    uint64_t disk_size = 0;
    TEST_ASSERT_OK(pio::FileSystem::GetFileSize(path, &disk_size));
    TEST_ASSERT(disk_size == expected_size);

    // Now test sequential read
    std::unique_ptr<pio::SequentialFile> rfile;
    TEST_ASSERT_OK(pio::FileSystem::NewSequentialFile(path, &rfile));

    std::vector<char> scratch(chunk1.size());
    pio::Slice result;
    TEST_ASSERT_OK(rfile->Read(chunk1.size(), &result, scratch.data()));
    TEST_ASSERT(result.ToString() == chunk1);

    // Skip chunk2
    TEST_ASSERT_OK(rfile->Skip(chunk2.size()));

    // Read start of large chunk
    std::vector<char> scratch_lg(100);
    TEST_ASSERT_OK(rfile->Read(100, &result, scratch_lg.data()));
    TEST_ASSERT(result.size() == 100);
    TEST_ASSERT(result.ToString() == std::string(100, 'X'));

    TEST_ASSERT_OK(rfile->Close());
    TEST_ASSERT_OK(pio::FileSystem::DeleteFile(path));
    std::cout << " PASSED" << std::endl;
}

void TestRandomAccessFile() {
    std::cout << "[TEST] TestRandomAccessFile..." << std::flush;
    std::string path = "pio_test_rnd.bin";
    (void)pio::FileSystem::DeleteFile(path);

    std::unique_ptr<pio::WritableFile> wfile;
    TEST_ASSERT_OK(pio::FileSystem::NewWritableFile(path, &wfile));

    std::string data = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
    TEST_ASSERT_OK(wfile->Append(pio::Slice(data)));
    TEST_ASSERT_OK(wfile->Close());

    std::unique_ptr<pio::RandomAccessFile> rfile;
    TEST_ASSERT_OK(pio::FileSystem::NewRandomAccessFile(path, &rfile));

    char scratch[64];
    pio::Slice result;

    // Read numbers at offset 0
    TEST_ASSERT_OK(rfile->Read(0, 10, &result, scratch));
    TEST_ASSERT(result.ToString() == "0123456789");

    // Read uppercase at offset 10
    TEST_ASSERT_OK(rfile->Read(10, 26, &result, scratch));
    TEST_ASSERT(result.ToString() == "ABCDEFGHIJKLMNOPQRSTUVWXYZ");

    // Read lowercase at offset 36
    TEST_ASSERT_OK(rfile->Read(36, 26, &result, scratch));
    TEST_ASSERT(result.ToString() == "abcdefghijklmnopqrstuvwxyz");

    // Read near EOF
    TEST_ASSERT_OK(rfile->Read(60, 10, &result, scratch));
    TEST_ASSERT(result.size() == 2);
    TEST_ASSERT(result.ToString() == "yz");

    // Read at EOF
    TEST_ASSERT_OK(rfile->Read(data.size(), 10, &result, scratch));
    TEST_ASSERT(result.empty());

    TEST_ASSERT_OK(rfile->Close());
    TEST_ASSERT_OK(pio::FileSystem::DeleteFile(path));
    std::cout << " PASSED" << std::endl;
}

void TestMemoryMappedFile() {
    std::cout << "[TEST] TestMemoryMappedFile..." << std::flush;
    std::string path = "pio_test_mmap.bin";
    (void)pio::FileSystem::DeleteFile(path);

    std::unique_ptr<pio::WritableFile> wfile;
    TEST_ASSERT_OK(pio::FileSystem::NewWritableFile(path, &wfile));

    std::string content = "Zero-Copy Memory-Mapped Vector Storage by PomaiDB/pio";
    TEST_ASSERT_OK(wfile->Append(pio::Slice(content)));
    TEST_ASSERT_OK(wfile->Close());

    // Test normal mapping
    {
        std::unique_ptr<pio::MemoryMappedFile> mmap_file;
        TEST_ASSERT_OK(pio::FileSystem::NewMemoryMappedFile(path, &mmap_file));
        TEST_ASSERT(mmap_file->size() == content.size());
        TEST_ASSERT(mmap_file->data() != nullptr);

        std::string mapped_str(reinterpret_cast<const char*>(mmap_file->data()), mmap_file->size());
        TEST_ASSERT(mapped_str == content);

        mmap_file->Advise(pio::Advice::Sequential);
        mmap_file->Advise(pio::Advice::WillNeed);
        mmap_file->Close();
        TEST_ASSERT(mmap_file->size() == 0);
        TEST_ASSERT(mmap_file->data() == nullptr);
    }

    TEST_ASSERT_OK(pio::FileSystem::DeleteFile(path));

    // Test empty file mapping
    std::string empty_path = "pio_test_empty.bin";
    {
        std::unique_ptr<pio::WritableFile> w_empty;
        TEST_ASSERT_OK(pio::FileSystem::NewWritableFile(empty_path, &w_empty));
        TEST_ASSERT_OK(w_empty->Close());

        std::unique_ptr<pio::MemoryMappedFile> mmap_empty;
        TEST_ASSERT_OK(pio::FileSystem::NewMemoryMappedFile(empty_path, &mmap_empty));
        TEST_ASSERT(mmap_empty->size() == 0);
        TEST_ASSERT(mmap_empty->data() == nullptr);
        mmap_empty->Close();
        TEST_ASSERT_OK(pio::FileSystem::DeleteFile(empty_path));
    }

    std::cout << " PASSED" << std::endl;
}

void TestAppendAndPositionalWrite() {
    std::cout << "[TEST] TestAppendAndPositionalWrite..." << std::flush;
    std::string path = "pio_test_append.bin";
    (void)pio::FileSystem::DeleteFile(path);

    // Initial write
    {
        std::unique_ptr<pio::WritableFile> wfile;
        TEST_ASSERT_OK(pio::FileSystem::NewWritableFile(path, &wfile));
        TEST_ASSERT_OK(wfile->Append(pio::Slice("AAAAA")));
        TEST_ASSERT_OK(wfile->Close());
    }

    // Append mode
    {
        std::unique_ptr<pio::WritableFile> wfile;
        TEST_ASSERT_OK(pio::FileSystem::NewAppendableFile(path, &wfile));
        TEST_ASSERT_OK(wfile->Append(pio::Slice("BBBBB")));
        TEST_ASSERT_OK(wfile->Close());
    }

    // Verify append
    {
        std::unique_ptr<pio::RandomAccessFile> rfile;
        TEST_ASSERT_OK(pio::FileSystem::NewRandomAccessFile(path, &rfile));
        char scratch[16];
        pio::Slice res;
        TEST_ASSERT_OK(rfile->Read(0, 10, &res, scratch));
        TEST_ASSERT(res.ToString() == "AAAAABBBBB");
        TEST_ASSERT_OK(rfile->Close());
    }

    // Positional overwrite (Pwrite)
    {
        std::unique_ptr<pio::WritableFile> wfile;
        TEST_ASSERT_OK(pio::FileSystem::NewWritableFile(path, &wfile));
        TEST_ASSERT_OK(wfile->Append(pio::Slice("0123456789")));
        TEST_ASSERT_OK(wfile->Pwrite(3, pio::Slice("XYZ")));
        TEST_ASSERT_OK(wfile->Close());

        std::unique_ptr<pio::RandomAccessFile> rfile;
        TEST_ASSERT_OK(pio::FileSystem::NewRandomAccessFile(path, &rfile));
        char scratch[16];
        pio::Slice res;
        TEST_ASSERT_OK(rfile->Read(0, 10, &res, scratch));
        TEST_ASSERT(res.ToString() == "012XYZ6789");
        TEST_ASSERT_OK(rfile->Close());
    }

    TEST_ASSERT_OK(pio::FileSystem::DeleteFile(path));
    std::cout << " PASSED" << std::endl;
}

void TestAtomicRenameAndDelete() {
    std::cout << "[TEST] TestAtomicRenameAndDelete..." << std::flush;
    std::string src = "pio_test_rename_src.tmp";
    std::string dst = "pio_test_rename_dst.tmp";
    (void)pio::FileSystem::DeleteFile(src);
    (void)pio::FileSystem::DeleteFile(dst);

    {
        std::unique_ptr<pio::WritableFile> wfile;
        TEST_ASSERT_OK(pio::FileSystem::NewWritableFile(src, &wfile));
        TEST_ASSERT_OK(wfile->Append(pio::Slice("Atomic-Rename-Payload")));
        TEST_ASSERT_OK(wfile->Close());
    }

    TEST_ASSERT(pio::FileSystem::FileExists(src));
    TEST_ASSERT(!pio::FileSystem::FileExists(dst));

    TEST_ASSERT_OK(pio::FileSystem::RenameFileAtomic(src, dst));
    TEST_ASSERT(!pio::FileSystem::FileExists(src));
    TEST_ASSERT(pio::FileSystem::FileExists(dst));

    // Verify content in dst
    {
        std::unique_ptr<pio::RandomAccessFile> rfile;
        TEST_ASSERT_OK(pio::FileSystem::NewRandomAccessFile(dst, &rfile));
        char scratch[32];
        pio::Slice res;
        TEST_ASSERT_OK(rfile->Read(0, 21, &res, scratch));
        TEST_ASSERT(res.ToString() == "Atomic-Rename-Payload");
        TEST_ASSERT_OK(rfile->Close());
    }

    TEST_ASSERT_OK(pio::FileSystem::DeleteFile(dst));
    TEST_ASSERT(!pio::FileSystem::FileExists(dst));
    std::cout << " PASSED" << std::endl;
}

void TestDirectoryOperations() {
    std::cout << "[TEST] TestDirectoryOperations..." << std::flush;
    std::string test_dir = "pio_test_directory_suite/subdir";
    (void)pio::FileSystem::CreateDirAll(test_dir);

    std::string file1 = test_dir + "/item1.txt";
    std::string file2 = test_dir + "/item2.txt";

    {
        std::unique_ptr<pio::WritableFile> w1, w2;
        TEST_ASSERT_OK(pio::FileSystem::NewWritableFile(file1, &w1));
        TEST_ASSERT_OK(w1->Append(pio::Slice("item1")));
        TEST_ASSERT_OK(w1->Close());

        TEST_ASSERT_OK(pio::FileSystem::NewWritableFile(file2, &w2));
        TEST_ASSERT_OK(w2->Append(pio::Slice("item2")));
        TEST_ASSERT_OK(w2->Close());
    }

    std::vector<std::string> entries;
    TEST_ASSERT_OK(pio::FileSystem::ListDirectory(test_dir, &entries));
    TEST_ASSERT(entries.size() == 2);
    bool found1 = false, found2 = false;
    for (const auto& e : entries) {
        if (e == "item1.txt") found1 = true;
        if (e == "item2.txt") found2 = true;
    }
    TEST_ASSERT(found1 && found2);

    TEST_ASSERT_OK(pio::FileSystem::SyncDir(test_dir));

    TEST_ASSERT_OK(pio::FileSystem::DeleteFile(file1));
    TEST_ASSERT_OK(pio::FileSystem::DeleteFile(file2));
    TEST_ASSERT_OK(pio::FileSystem::DeleteFile(test_dir));
    TEST_ASSERT_OK(pio::FileSystem::DeleteFile("pio_test_directory_suite"));
    std::cout << " PASSED" << std::endl;
}

int main() {
    std::cout << "======================================================" << std::endl;
    std::cout << "      PIO STORAGE & DIRECT I/O SUITE VERIFICATION     " << std::endl;
    std::cout << "======================================================" << std::endl;

    auto t0 = std::chrono::high_resolution_clock::now();

    TestSlice();
    TestStatus();
    TestDirectBuffer();
    TestWritableAndSequentialFile();
    TestRandomAccessFile();
    TestMemoryMappedFile();
    TestAppendAndPositionalWrite();
    TestAtomicRenameAndDelete();
    TestDirectoryOperations();

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "======================================================" << std::endl;
    std::cout << " ALL PIO TEST SUITES PASSED CLEANLY in " << elapsed << "s" << std::endl;
    std::cout << "======================================================" << std::endl;
    return 0;
}

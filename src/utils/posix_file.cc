#include "posix_file.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>

#if defined(_WIN32) || defined(_WIN64)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#ifndef O_BINARY
#define O_BINARY _O_BINARY
#endif
#else
#include <unistd.h>
#include <sys/mman.h>
#ifndef O_BINARY
#define O_BINARY 0
#endif
#endif

namespace pomai::util
{

    static pomai::Status Err(std::string_view what)
    {
        return pomai::Status::IoError(std::string(what) + ": " + std::strerror(errno));
    }

#if defined(_WIN32) || defined(_WIN64)
    static ssize_t pread_compat(int fd, void* buf, size_t count, off_t offset) {
        HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
        if (h == INVALID_HANDLE_VALUE) return -1;
        OVERLAPPED ov{};
        ov.Offset = static_cast<DWORD>(static_cast<uint64_t>(offset) & 0xFFFFFFFF);
        ov.OffsetHigh = static_cast<DWORD>((static_cast<uint64_t>(offset) >> 32) & 0xFFFFFFFF);
        DWORD read_bytes = 0;
        if (!ReadFile(h, buf, static_cast<DWORD>(count), &read_bytes, &ov)) return -1;
        return static_cast<ssize_t>(read_bytes);
    }

    static ssize_t pwrite_compat(int fd, const void* buf, size_t count, off_t offset) {
        HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
        if (h == INVALID_HANDLE_VALUE) return -1;
        OVERLAPPED ov{};
        ov.Offset = static_cast<DWORD>(static_cast<uint64_t>(offset) & 0xFFFFFFFF);
        ov.OffsetHigh = static_cast<DWORD>((static_cast<uint64_t>(offset) >> 32) & 0xFFFFFFFF);
        DWORD written = 0;
        if (!WriteFile(h, buf, static_cast<DWORD>(count), &written, &ov)) return -1;
        return static_cast<ssize_t>(written);
    }
#endif

    PosixFile::PosixFile(PosixFile &&other) noexcept : fd_(other.fd_)
    {
        other.fd_ = -1;
    }

    PosixFile &PosixFile::operator=(PosixFile &&other) noexcept
    {
        if (this != &other)
        {
            (void)Close();
            fd_ = other.fd_;
            other.fd_ = -1;
        }
        return *this;
    }

    PosixFile::~PosixFile() { (void)Close(); }

    pomai::Status PosixFile::OpenAppend(const std::string &path, PosixFile *out)
    {
        int fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_BINARY, 0644);
        if (fd < 0)
            return Err("open");
        *out = PosixFile(fd);
        return pomai::Status::Ok();
    }

    pomai::Status PosixFile::OpenRead(const std::string &path, PosixFile *out)
    {
        int fd = ::open(path.c_str(), O_RDONLY | O_BINARY);
        if (fd < 0)
            return Err("open");
        *out = PosixFile(fd);
        return pomai::Status::Ok();
    }

    pomai::Status PosixFile::CreateTrunc(const std::string &path, PosixFile *out)
    {
        int fd = ::open(path.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_BINARY, 0644);
        if (fd < 0)
            return Err("open");
        *out = PosixFile(fd);
        return pomai::Status::Ok();
    }

    pomai::Status PosixFile::PWrite(std::uint64_t off, const void *data, std::size_t n)
    {
        const std::uint8_t *p = static_cast<const std::uint8_t *>(data);
        std::size_t done = 0;
        while (done < n)
        {
#if defined(_WIN32) || defined(_WIN64)
            ssize_t w = pwrite_compat(fd_, p + done, n - done, static_cast<off_t>(off + done));
#else
            ssize_t w = ::pwrite(fd_, p + done, n - done, static_cast<off_t>(off + done));
#endif
            if (w < 0)
            {
                if (errno == EINTR)
                    continue;
                return Err("pwrite");
            }
            done += static_cast<std::size_t>(w);
        }
        return pomai::Status::Ok();
    }

    pomai::Status PosixFile::ReadAt(std::uint64_t off, void *data, std::size_t n, std::size_t *out_read)
    {
        std::uint8_t *p = static_cast<std::uint8_t *>(data);
        std::size_t done = 0;
        while (done < n)
        {
#if defined(_WIN32) || defined(_WIN64)
            ssize_t r = pread_compat(fd_, p + done, n - done, static_cast<off_t>(off + done));
#else
            ssize_t r = ::pread(fd_, p + done, n - done, static_cast<off_t>(off + done));
#endif
            if (r < 0)
            {
                if (errno == EINTR)
                    continue;
                return Err("pread");
            }
            if (r == 0)
                break;
            done += static_cast<std::size_t>(r);
        }
        *out_read = done;
        return pomai::Status::Ok();
    }

    pomai::Status PosixFile::Flush() { return pomai::Status::Ok(); }

    pomai::Status PosixFile::SyncData()
    {
#if defined(_WIN32) || defined(_WIN64)
        if (_commit(fd_) != 0)
            return Err("commit");
#else
        if (::fdatasync(fd_) != 0)
            return Err("fdatasync");
#endif
        return pomai::Status::Ok();
    }


    pomai::Status PosixFile::SyncAll()
    {
#if defined(_WIN32) || defined(_WIN64)
        if (_commit(fd_) != 0)
            return Err("commit");
#else
        if (::fsync(fd_) != 0)
            return Err("fsync");
#endif
        return pomai::Status::Ok();
    }

    pomai::Status PosixFile::Map(const void** out_data, std::size_t* out_size)
    {
         if (fd_ < 0) return pomai::Status::InvalidArgument("file not open");
         if (map_addr_) {
             // Already mapped
             *out_data = map_addr_;
             *out_size = map_size_;
             return pomai::Status::Ok(); 
         }

         struct stat st;
         if (fstat(fd_, &st) != 0) return Err("fstat");
         std::size_t size = static_cast<std::size_t>(st.st_size);
         
         if (size == 0) {
             // Empty file, mapping might fail or return nothing
             map_addr_ = nullptr;
             map_size_ = 0;
             *out_data = nullptr;
             *out_size = 0;
             return pomai::Status::Ok();
         }

#if defined(_WIN32) || defined(_WIN64)
         HANDLE h = reinterpret_cast<HANDLE>(_get_osfhandle(fd_));
         if (h == INVALID_HANDLE_VALUE) return Err("get_osfhandle");
         HANDLE h_map = CreateFileMappingA(h, NULL, PAGE_READONLY, 0, 0, NULL);
         if (!h_map) return Err("CreateFileMapping");
         void* addr = MapViewOfFile(h_map, FILE_MAP_READ, 0, 0, size);
         CloseHandle(h_map);
         if (!addr) return Err("MapViewOfFile");
#else
         void* addr = ::mmap(nullptr, size, PROT_READ, MAP_SHARED, fd_, 0);
         if (addr == MAP_FAILED) return Err("mmap");
#endif
         
         map_addr_ = addr;
         map_size_ = size;
         *out_data = addr;
         *out_size = size;
         return pomai::Status::Ok();
    }

    pomai::Status PosixFile::Close()
    {
        if (map_addr_) {
#if defined(_WIN32) || defined(_WIN64)
            UnmapViewOfFile(map_addr_);
#else
            ::munmap(map_addr_, map_size_);
#endif
            map_addr_ = nullptr;
            map_size_ = 0;
        }

        if (fd_ >= 0)
        {
            int r;
            do
            {
                r = ::close(fd_);
            } while (r != 0 && errno == EINTR);
            fd_ = -1;
            if (r != 0)
                return Err("close");
        }
        return pomai::Status::Ok();
    }

    pomai::Status FsyncDir(const std::string &dir_path)
    {
#if defined(_WIN32) || defined(_WIN64)
        (void)dir_path;
        return pomai::Status::Ok();
#else
        int fd = ::open(dir_path.c_str(), O_DIRECTORY | O_RDONLY);
        if (fd < 0)
            return Err("open dir");
        int r;
        do
        {
            r = ::fsync(fd);
        } while (r != 0 && errno == EINTR);
        int saved = errno;
        ::close(fd);
        errno = saved;
        if (r != 0)
            return Err("fsync dir");
        return pomai::Status::Ok();
#endif
    }

} // namespace pomai::util
#pragma once

#include "ggml.h" // for ggml_log_level

#include <string>
#include <vector>

#ifdef __GNUC__
#    if defined(__MINGW32__) && !defined(__clang__)
#        define CHEESE_ATTRIBUTE_FORMAT(...) __attribute__((format(gnu_printf, __VA_ARGS__)))
#    else
#        define CHEESE_ATTRIBUTE_FORMAT(...) __attribute__((format(printf, __VA_ARGS__)))
#    endif
#else
#    define CHEESE_ATTRIBUTE_FORMAT(...)
#endif

//
// logging
//

CHEESE_ATTRIBUTE_FORMAT(2, 3)
void cheese_log_internal        (ggml_log_level level, const char * format, ...);
void cheese_log_callback_default(ggml_log_level level, const char * text, void * user_data);

#define CHEESE_LOG(...)       cheese_log_internal(GGML_LOG_LEVEL_NONE , __VA_ARGS__)
#define CHEESE_LOG_INFO(...)  cheese_log_internal(GGML_LOG_LEVEL_INFO , __VA_ARGS__)
#define CHEESE_LOG_WARN(...)  cheese_log_internal(GGML_LOG_LEVEL_WARN , __VA_ARGS__)
#define CHEESE_LOG_ERROR(...) cheese_log_internal(GGML_LOG_LEVEL_ERROR, __VA_ARGS__)
#define CHEESE_LOG_DEBUG(...) cheese_log_internal(GGML_LOG_LEVEL_DEBUG, __VA_ARGS__)
#define CHEESE_LOG_CONT(...)  cheese_log_internal(GGML_LOG_LEVEL_CONT , __VA_ARGS__)

//
// helpers
//

template <typename T>
struct no_init {
    T value;
    no_init() = default;
};

struct time_meas {
    time_meas(int64_t & t_acc, bool disable = false);
    ~time_meas();

    const int64_t t_start_us;

    int64_t & t_acc;
};

template <typename T>
struct buffer_view {
    T * data;
    size_t size = 0;

    bool has_data() const {
        return data && size > 0;
    }
};

void replace_all(std::string & s, const std::string & search, const std::string & replace);

// TODO: rename to cheese_format ?
CHEESE_ATTRIBUTE_FORMAT(1, 2)
std::string format(const char * fmt, ...);

std::string cheese_format_tensor_shape(const std::vector<int64_t> & ne);
std::string cheese_format_tensor_shape(const struct ggml_tensor * t);

std::string gguf_kv_to_str(const struct gguf_context * ctx_gguf, int i);

#define CHEESE_TENSOR_NAME_FATTN "__fattn__"

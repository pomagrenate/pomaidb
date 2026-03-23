#include "cheese-memory.h"

cheese_memory_status cheese_memory_status_combine(cheese_memory_status s0, cheese_memory_status s1) {
    bool has_update = false;

    switch (s0) {
        case CHEESE_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case CHEESE_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case CHEESE_MEMORY_STATUS_FAILED_PREPARE:
        case CHEESE_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s0;
            }
    }

    switch (s1) {
        case CHEESE_MEMORY_STATUS_SUCCESS:
            {
                has_update = true;
                break;
            }
        case CHEESE_MEMORY_STATUS_NO_UPDATE:
            {
                break;
            }
        case CHEESE_MEMORY_STATUS_FAILED_PREPARE:
        case CHEESE_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return s1;
            }
    }

    // if either status has an update, then the combined status has an update
    return has_update ? CHEESE_MEMORY_STATUS_SUCCESS : CHEESE_MEMORY_STATUS_NO_UPDATE;
}

bool cheese_memory_status_is_fail(cheese_memory_status status) {
    switch (status) {
        case CHEESE_MEMORY_STATUS_SUCCESS:
        case CHEESE_MEMORY_STATUS_NO_UPDATE:
            {
                return false;
            }
        case CHEESE_MEMORY_STATUS_FAILED_PREPARE:
        case CHEESE_MEMORY_STATUS_FAILED_COMPUTE:
            {
                return true;
            }
    }

    return false;
}

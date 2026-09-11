
# Fuzz targets for PomaiDB
if(ENABLE_FUZZING)
    set(FUZZER_FLAGS "-fsanitize=fuzzer,address,undefined")
    
    macro(add_pomai_fuzzer target source)
        add_executable(${target} ${source})
        target_compile_options(${target} PRIVATE ${FUZZER_FLAGS})
        target_link_options(${target} PRIVATE ${FUZZER_FLAGS})
        target_link_libraries(${target} PRIVATE pomai)
        target_include_directories(${target} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../src
            ${CMAKE_CURRENT_SOURCE_DIR}/../../src/capi
            ${CMAKE_CURRENT_SOURCE_DIR}/../../src/utils
        )
    endmacro()

    add_pomai_fuzzer(membrane_fuzzer membrane_fuzzer.cc)
    add_pomai_fuzzer(storage_fuzzer storage_fuzzer.cc)
endif()

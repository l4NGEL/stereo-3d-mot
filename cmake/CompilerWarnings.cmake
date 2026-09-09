# Shared warning configuration applied to every first-party target.
# Usage:  s3m_set_target_warnings(<target>)

function(s3m_set_target_warnings target)
    set(gcc_clang_flags
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -Wnon-virtual-dtor
        -Wunused
        -Woverloaded-virtual
        -Wformat=2)

    set(msvc_flags /W4 /permissive-)

    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target} PRIVATE ${gcc_clang_flags})
        if(S3M_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
    elseif(MSVC)
        target_compile_options(${target} PRIVATE ${msvc_flags})
        if(S3M_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE /WX)
        endif()
    endif()
endfunction()

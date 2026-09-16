# Cross-compile an x64 Windows build from Linux with clang-cl and an xwin
# MSVC/Windows SDK installation. This is a compile/link gate; produced binaries
# are not executed on the Linux host and CUDA remains a native-Windows gate.

set(CMAKE_SYSTEM_NAME Windows)
set(CMAKE_SYSTEM_PROCESSOR AMD64)
set(CMAKE_TRY_COMPILE_CONFIGURATION Release)

if(NOT DEFINED ENV{XWIN_ROOT} OR "$ENV{XWIN_ROOT}" STREQUAL "")
    message(FATAL_ERROR "XWIN_ROOT must name the xwin splat directory")
endif()
file(TO_CMAKE_PATH "$ENV{XWIN_ROOT}" XWIN_ROOT)

set(CMAKE_C_COMPILER   clang-cl)
set(CMAKE_CXX_COMPILER clang-cl)
set(CMAKE_C_COMPILER_TARGET   x86_64-pc-windows-msvc)
set(CMAKE_CXX_COMPILER_TARGET x86_64-pc-windows-msvc)
set(CMAKE_AR          llvm-lib)
set(CMAKE_LINKER      lld-link)
set(CMAKE_RC_COMPILER llvm-rc)
set(CMAKE_MT          llvm-mt)

set(_xwin_compile_flags
    "-fuse-ld=lld /vctoolsdir \"${XWIN_ROOT}/crt\" /winsdkdir \"${XWIN_ROOT}/sdk\"")
set(CMAKE_C_FLAGS_INIT   "${_xwin_compile_flags}")
set(CMAKE_CXX_FLAGS_INIT "${_xwin_compile_flags}")

set(_xwin_link_flags
    "/libpath:\"${XWIN_ROOT}/crt/lib/x86_64\" /libpath:\"${XWIN_ROOT}/sdk/lib/ucrt/x86_64\" /libpath:\"${XWIN_ROOT}/sdk/lib/um/x86_64\"")
set(CMAKE_EXE_LINKER_FLAGS_INIT    "${_xwin_link_flags}")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "${_xwin_link_flags}")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "${_xwin_link_flags}")

unset(_xwin_compile_flags)
unset(_xwin_link_flags)

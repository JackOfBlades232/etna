# Unlike FindVulkan.cmake, no FindSlang.cmake is shipped with the SDK

find_program(SLANGC_EXECUTABLE
  NAMES slangc
  HINTS "$ENV{VULKAN_SDK}" "${SLANG_SDK_DIR}"
  PATH_SUFFIXES bin Bin
  DOC "Path to the slangc shader compiler"
)

find_path(SLANG_INCLUDE_DIR
  NAMES slang/slang.h
  HINTS "$ENV{VULKAN_SDK}" "${SLANG_SDK_DIR}"
  PATH_SUFFIXES include Include
  DOC "Directory containing slang.h"
)

find_library(SLANG_LIBRARY
  NAMES slang
  HINTS "$ENV{VULKAN_SDK}" "${SLANG_SDK_DIR}"
  PATH_SUFFIXES lib Lib
  DOC "Slang link library (.lib import lib on Windows, .so/.dylib elsewhere)"
)

if(WIN32)
  find_file(SLANG_RUNTIME_LIB
    NAMES slang.dll
    HINTS "$ENV{VULKAN_SDK}" "${SLANG_SDK_DIR}"
    PATH_SUFFIXES bin Bin
    DOC "Slang runtime DLL"
  )
endif()

if(SLANGC_EXECUTABLE)
  execute_process(
    COMMAND "${SLANGC_EXECUTABLE}" -v
    OUTPUT_VARIABLE SLANG_VERSION_RAW
    ERROR_VARIABLE  SLANG_VERSION_RAW # slangc may print to stderr instead of stdout
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  # SLANG_VERSION_RAW looks like "v2024.14.5" or "v2024.14.5-3-gabc1234"
  if(SLANG_VERSION_RAW MATCHES "([0-9]+\\.[0-9]+\\.[0-9]+)")
    set(SLANG_VERSION "${CMAKE_MATCH_1}")
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Slang
  REQUIRED_VARS SLANGC_EXECUTABLE SLANG_INCLUDE_DIR SLANG_LIBRARY
  VERSION_VAR SLANG_VERSION
)

if(Slang_FOUND AND NOT TARGET Slang::slangc)
  add_executable(Slang::slangc IMPORTED)
  set_target_properties(Slang::slangc PROPERTIES
    IMPORTED_LOCATION "${SLANGC_EXECUTABLE}"
  )
endif()

if(Slang_FOUND AND NOT TARGET Slang::slang)
  add_library(Slang::slang SHARED IMPORTED)
  set_target_properties(Slang::slang PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${SLANG_INCLUDE_DIR}"
  )
  if(WIN32)
    set_target_properties(Slang::slang PROPERTIES
      IMPORTED_LOCATION   "${SLANG_RUNTIME_LIB}"  # .dll
      IMPORTED_IMPLIB     "${SLANG_LIBRARY}"      # .lib
    )
  else()
    set_target_properties(Slang::slang PROPERTIES
      IMPORTED_LOCATION "${SLANG_LIBRARY}"        # .so / .dylib
    )
  endif()
endif()

mark_as_advanced(SLANGC_EXECUTABLE SLANG_INCLUDE_DIR SLANG_LIBRARY SLANG_RUNTIME_LIB)

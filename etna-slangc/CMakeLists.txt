cmake_minimum_required(VERSION 3.25)

add_executable(etna-slangc
  "source/main.cpp"
  "source/Slang.cpp")

target_include_directories(etna-slangc PRIVATE source)

target_link_libraries(etna-slangc PUBLIC
  Slang::slang
  spdlog::spdlog
)

# @TODO: move to common code
if (${ETNA_DEBUG})
  target_compile_definitions(etna-slangc PUBLIC ETNA_DEBUG=1)
endif ()

if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC") # cl and clang-cl
  target_compile_options(etna-slangc PRIVATE /W4 /WX
    # Selectively disable some insane warnings.
    /wd4061 /wd4514
    # Enforce standards-compliance in MSVC
    /permissive- /volatile:iso /Zc:inline /Zc:wchar_t /EHsc /Zc:__cplusplus
  )
  if (CMAKE_CXX_COMPILER_ID MATCHES "MSVC") # cl only
    target_compile_options(etna-slangc PRIVATE /Zc:preprocessor)
  endif()
elseif(CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "GNU") # gcc and clang
  target_compile_options(etna-slangc PRIVATE -Wall -Wextra -Werror -pedantic)
endif()

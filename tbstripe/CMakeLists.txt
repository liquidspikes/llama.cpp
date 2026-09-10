cmake_minimum_required(VERSION 3.14)
project(tbstripe C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Threads REQUIRED)

include_directories(include)

add_library(tbstripe STATIC
    src/tbstripe.c
    src/stripe.c
    src/sysfs.c
)
target_link_libraries(tbstripe PRIVATE Threads::Threads m)

add_library(tbstripe_shared SHARED
    src/tbstripe.c
    src/stripe.c
    src/sysfs.c
)
set_target_properties(tbstripe_shared PROPERTIES OUTPUT_NAME "tbstripe")
target_link_libraries(tbstripe_shared PRIVATE Threads::Threads m)

add_executable(tbstripe-bench tools/tbstripe-bench.c)
target_link_libraries(tbstripe-bench PRIVATE tbstripe Threads::Threads m)

add_executable(tb-hrx-xchg tools/tb-hrx-xchg.c)
target_link_libraries(tb-hrx-xchg PRIVATE tbstripe Threads::Threads m ${CMAKE_DL_LIBS})

install(TARGETS tbstripe tbstripe_shared tbstripe-bench tb-hrx-xchg
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES include/tbstripe.h DESTINATION include)

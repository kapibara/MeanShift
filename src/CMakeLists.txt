
set(TEST exec)
set(MEANSHIFT meanshift)

set(LIBHEADERS meanshift.h)

set(LIBSOURCE meanshift.cpp)

add_executable(${TEST} main.cpp ${LIBSOURCE} ${LIBHEADERS})
add_library(${MEANSHIFT} ${LIBSOURCE} ${LIBHEADERS})

message(STATUS "${FLANN_LIBS}")

target_link_libraries(${TEST} ${FLANN_LIBS})
target_link_libraries(${MEANSHIFT}) #no flann libs required

install(TARGETS ${TEST} DESTINATION .)
install(TARGETS ${MEANSHIFT} DESTINATION ../lib)
    

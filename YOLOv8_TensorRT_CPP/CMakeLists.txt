set(Target YOLOv8_TensorRT_CPP)

add_library(${Target} SHARED 
    ${CMAKE_CURRENT_LIST_DIR}/include/yolov8.h 
    ${CMAKE_CURRENT_LIST_DIR}/src/yolov8.cpp 
)

target_include_directories(${Target} 
PUBLIC  
    ${CMAKE_CURRENT_LIST_DIR}/include 
)

target_link_libraries(${Target} 
PUBLIC 
    TensorRT_CPP_API 
)

target_compile_definitions(${Target} PRIVATE YOLOV8_TENSORRT_CPP_EXPORT)

set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR}/install)

install(TARGETS ${Target}
        RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
        ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)

install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})
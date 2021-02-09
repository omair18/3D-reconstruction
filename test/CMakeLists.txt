add_executable(http_server_test http_server_test.cpp)
target_link_libraries(http_server_test Boost::system pthread ssl crypto)

add_executable(test test.cpp)

add_executable(test_json test_json.cpp)
target_link_libraries(test_json Boost::json)

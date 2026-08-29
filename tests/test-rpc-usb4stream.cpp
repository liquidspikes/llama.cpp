#include "ggml-rpc.h"
#include "transport.h"
#include "transport-usb4stream.h"
#include "ggml.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#define TEST_CHECK(cond) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "TEST FAILURE: %s at %s:%d\n", #cond, __FILE__, __LINE__); \
            std::exit(1); \
        } \
    } while (0)

static void test_endpoint_detection() {
    printf("[test_endpoint_detection] Testing endpoint parsing and detection...\n");

    TEST_CHECK(socket_t::is_usb4_endpoint("usb4:/dev/tbstream0"));
    TEST_CHECK(socket_t::is_usb4_endpoint("usb4:0"));
    TEST_CHECK(socket_t::is_usb4_endpoint("/dev/tbstream0"));
    TEST_CHECK(socket_t::is_usb4_endpoint("/dev/tbstream12"));
    TEST_CHECK(socket_t::is_usb4_endpoint("tbstream:0"));
    TEST_CHECK(socket_t::is_usb4_endpoint("mock_usb4_pipe"));
    TEST_CHECK(!socket_t::is_usb4_endpoint("127.0.0.1:50052"));
    TEST_CHECK(!socket_t::is_usb4_endpoint("192.168.1.100:8080"));
    TEST_CHECK(!socket_t::is_usb4_endpoint("localhost:50052"));

    TEST_CHECK(socket_t::normalize_usb4_dev_path("usb4:/dev/tbstream0") == "/dev/tbstream0");
    TEST_CHECK(socket_t::normalize_usb4_dev_path("usb4:0") == "/dev/tbstream0");
    TEST_CHECK(socket_t::normalize_usb4_dev_path("usb4://1") == "/dev/tbstream1");
    TEST_CHECK(socket_t::normalize_usb4_dev_path("/dev/tbstream2") == "/dev/tbstream2");

    printf("[test_endpoint_detection] Passed!\n");
}

static void test_loopback_transport_data_transfer() {
    printf("[test_loopback_transport_data_transfer] Testing direct USB4 transport stream...\n");

    auto pair = usb4stream_transport::create_loopback_pair();
    auto & tx = pair.first;
    auto & rx = pair.second;

    TEST_CHECK(tx != nullptr);
    TEST_CHECK(rx != nullptr);

    // 1. Small message
    const char * msg = "Hello USB4Stream LLM Cluster!";
    size_t len = strlen(msg) + 1;

    TEST_CHECK(tx->send_data(msg, len));
    TEST_CHECK(tx->flush());

    char buf[128] = {};
    TEST_CHECK(rx->recv_data(buf, len));
    TEST_CHECK(strcmp(buf, msg) == 0);

    // 2. Large data chunk transfer (4 MiB)
    const size_t large_sz = 4 * 1024 * 1024;
    std::vector<uint8_t> send_buf(large_sz);
    std::vector<uint8_t> recv_buf(large_sz, 0);

    for (size_t i = 0; i < large_sz; ++i) {
        send_buf[i] = static_cast<uint8_t>(i & 0xFF);
    }

    auto fut = std::async(std::launch::async, [&]() {
        return rx->recv_data(recv_buf.data(), large_sz);
    });

    TEST_CHECK(tx->send_data(send_buf.data(), large_sz));
    TEST_CHECK(tx->flush());

    TEST_CHECK(fut.get());
    TEST_CHECK(memcmp(send_buf.data(), recv_buf.data(), large_sz) == 0);

    printf("[test_loopback_transport_data_transfer] Passed (transferred 4 MiB over USB4 loopback stream)!\n");
}

static void test_socket_usb4_server_client() {
    printf("[test_socket_usb4_server_client] Testing socket_t USB4 server / client abstraction...\n");

    const char * endpoint = "mock_usb4_cluster_0";

    auto server_socket = socket_t::create_server(endpoint, 0);
    TEST_CHECK(server_socket != nullptr);
    TEST_CHECK(server_socket->is_usb4());

    auto client_socket = socket_t::connect(endpoint, 0);
    TEST_CHECK(client_socket != nullptr);
    TEST_CHECK(client_socket->is_usb4());

    auto accepted_socket = server_socket->accept();
    TEST_CHECK(accepted_socket != nullptr);

    // Send command and tensor metadata simulation
    uint32_t send_cmd = 0x42;
    uint64_t tensor_id = 99991;
    std::vector<float> tensor_data(1024, 3.14159f);

    TEST_CHECK(client_socket->send_data(&send_cmd, sizeof(send_cmd)));
    TEST_CHECK(client_socket->send_data(&tensor_id, sizeof(tensor_id)));
    TEST_CHECK(client_socket->send_data(tensor_data.data(), tensor_data.size() * sizeof(float)));
    TEST_CHECK(client_socket->flush());

    uint32_t recv_cmd = 0;
    uint64_t recv_tensor_id = 0;
    std::vector<float> recv_tensor_data(1024, 0.0f);

    TEST_CHECK(accepted_socket->recv_data(&recv_cmd, sizeof(recv_cmd)));
    TEST_CHECK(accepted_socket->recv_data(&recv_tensor_id, sizeof(recv_tensor_id)));
    TEST_CHECK(accepted_socket->recv_data(recv_tensor_data.data(), recv_tensor_data.size() * sizeof(float)));

    TEST_CHECK(recv_cmd == send_cmd);
    TEST_CHECK(recv_tensor_id == tensor_id);
    for (size_t i = 0; i < tensor_data.size(); ++i) {
        TEST_CHECK(tensor_data[i] == recv_tensor_data[i]);
    }

    printf("[test_socket_usb4_server_client] Passed!\n");
}

int main() {
    printf("=== USB4STREAM Clustering Transport Tests ===\n");
    test_endpoint_detection();
    test_loopback_transport_data_transfer();
    test_socket_usb4_server_client();
    printf("=== All USB4STREAM Tests Passed Successfully ===\n");
    return 0;
}

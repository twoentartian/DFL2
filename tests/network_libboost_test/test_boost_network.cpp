#include <network.hpp>

#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE (boost_network_test)
	
	BOOST_AUTO_TEST_CASE (simple_client_server_header)
	{
		using namespace network;
		constexpr bool ENABLE_COUT = false;
		constexpr int PORT = 1500;
		constexpr uint32_t dataLength = 10 * 1024;
		
		std::atomic_int accept_counter = 0, ping_pong_counter = 0, clients_connect_counter = 0, clients_connect_fail_counter = 0;
		
		std::recursive_mutex cout_lock;
		simple::tcp_server_with_header server;
		
		server.SetAcceptHandler([&cout_lock, &accept_counter](const std::string &ip, uint16_t port, std::shared_ptr<simple::tcp_session> session)
		                        {
			                        if (ENABLE_COUT)
			                        {
				                        std::lock_guard<std::recursive_mutex> temp_lock_guard(cout_lock);
				                        std::cout << "[server] accept: " << ip << ":" << port << std::endl;
									}
			                        accept_counter++;
		                        });
		server.SetSessionReceiveHandler_with_header([&cout_lock, &ping_pong_counter](header::COMMAND_TYPE command, std::shared_ptr<std::string> data, std::shared_ptr<simple::tcp_session_with_header> session_receive)
		                                            {
			                                            ping_pong_counter++;
			                                            if (ENABLE_COUT)
			                                            {
				                                            std::lock_guard<std::recursive_mutex> temp_lock_guard(cout_lock);
				                                            std::cout << "[server] receive (ok) length " << data->length() << std::endl;
			                                            }
			                                            session_receive->write_with_header(0, data->data(), data->length());
		                                            });
		server.SetSessionCloseHandler([&](std::shared_ptr<simple::tcp_session> session_close)
		                              {
			                              if (ENABLE_COUT)
			                              {
				                              std::lock_guard<std::recursive_mutex> temp_lock_guard(cout_lock);
				                              std::cout << "[server] close: " << session_close->ip() << ":" << session_close->port() << std::endl;
			                              }
		                              });
		
		auto ret_code = server.Start(PORT, 6);
		std::cout << "server: " << std::to_string(ret_code) << std::endl;
		
		// tcp_client
		const int NumberOfClient = 1000;
		std::shared_ptr<simple::tcp_client_with_header> clients[NumberOfClient];
		int client_counts[NumberOfClient];
		
		for (size_t i = 0; i < NumberOfClient; i++)
		{
			clients[i] = simple::tcp_client_with_header::CreateClient();
			client_counts[i] = 0;
		}
		
		for (size_t i = 0; i < NumberOfClient; i++)
		{
			clients[i]->connect("127.0.0.1", PORT);
			clients[i]->SetConnectHandler([&clients_connect_counter, &clients_connect_fail_counter](tcp_status status, std::shared_ptr<simple::tcp_client> client)
			                              {
//				                              BOOST_CHECK(status == tcp_status::Success);
				                              auto *data = new uint8_t[dataLength];
				                              for (size_t i = 0; i < dataLength; i++)
				                              {
					                              data[i] = uint8_t(i);
				                              }
				                              auto send_status = std::static_pointer_cast<simple::tcp_client_with_header>(client)->write_with_header(1, data, dataLength);
				                              if (send_status == network::Success)
				                              {
					                              clients_connect_counter++;
				                              }
				                              else
				                              {
					                              clients_connect_fail_counter++;
				                              }
				
				                              delete[] data;
			                              });
			
			clients[i]->SetReceiveHandler_with_header([&cout_lock, i, &client_counts](header::COMMAND_TYPE command, std::shared_ptr<std::string> data, std::shared_ptr<simple::tcp_client_with_header> client)
			                                          {
				                                          if (ENABLE_COUT)
				                                          {
					                                          std::lock_guard<std::recursive_mutex> temp_lock_guard(cout_lock);
					                                          std::cout << "[client] receive (ok) length " << data->length() << " count: " << client_counts[i] << std::endl;
				                                          }
				                                          client_counts[i]++;
														  client->write_with_header(1, data->data(), data->length());
			                                          });
			
			clients[i]->SetCloseHandler([&cout_lock](const std::string &ip, uint16_t port)
			                            {
				                            if (ENABLE_COUT)
				                            {
					                            std::lock_guard<std::recursive_mutex> temp_lock_guard(cout_lock);
					                            std::cout << "[client] connection close: " << ip << ":" << port
					                                      << std::endl;//NEVER Delete this line, because when you delete it, the application will crash when close because the compiler will ignore this lambda and cause empty implementation exception.
				                            }
			                            });
		}
		
		std::this_thread::sleep_for(std::chrono::seconds(3));
		int ping_pong_previous = ping_pong_counter;
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
		for (size_t i = 0; i < NumberOfClient; i++)
		{
			clients[i]->Disconnect();
		}
		
		BOOST_TEST(accept_counter == NumberOfClient, "accept_counter = " << accept_counter);
		BOOST_TEST(clients_connect_counter == NumberOfClient, "clients_connect_counter = " << clients_connect_counter);
		BOOST_TEST(ping_pong_counter > ping_pong_previous, "server/client ping pong test. end: " << ping_pong_counter << "  previous:" << ping_pong_previous);
		BOOST_TEST(clients_connect_fail_counter == 0, "clients_connect_fail_counter = " << clients_connect_fail_counter);
		
		std::this_thread::sleep_for(std::chrono::milliseconds(500));
		server.Stop();
	}
	
	BOOST_AUTO_TEST_CASE (simple_p2p)
	{
		constexpr int PORT = 1600;
		network::p2p peers[2];
		std::string data = "01234567890123456789";
		
		std::atomic_int receive_count = 0;
		peers[1].set_receive_callback([&receive_count](const char* data, size_t length, std::string ip) -> std::string {
			receive_count++;
			return data;
		});
		auto status = peers[1].start_service(PORT, 6);
		BOOST_CHECK(status == network::i_p2p_node::success);
		
		constexpr int ITER = 1600;
		{
			for (int i = 0; i < ITER; ++i)
			{
				peers[0].send("127.0.0.1", PORT, network::i_p2p_node::ipv4, data.data(), data.size(),
				              [](network::i_p2p_node::send_packet_status status, const char* data, size_t size){
					              if (status != network::i_p2p_node::send_packet_success)
					              {
						              std::cout << "error: " << status << "\t" << network::i_p2p_node::send_packet_status_message[status] << std::endl;
								  }
				              });
			}
		}
		
		std::this_thread::sleep_for(std::chrono::seconds(1));
		std::cout << "receive_count is " << receive_count << ", should be " << ITER << std::endl;
		BOOST_CHECK(ITER == receive_count);
		
		peers[0].stop_service();
		peers[1].stop_service();
	}
	
	BOOST_AUTO_TEST_CASE (simple_p2p_header)
	{
		constexpr int PORT = 1700;
		network::p2p_with_header peers[2];
		const std::string data = "01234567890123456789";
		
		std::atomic_int receive_count = 0;
		peers[1].set_receive_callback([&receive_count](network::header::COMMAND_TYPE command, const char* data, size_t length, std::string ip)
		-> std::tuple<network::header::COMMAND_TYPE, std::string> {
			receive_count++;
			return {1, std::string(data, length)};
		});
		auto status = peers[1].start_service(PORT, 6);
		BOOST_CHECK(status == network::i_p2p_node_with_header::success);
		
		constexpr int ITER = 1600;
		{
			for (int i = 0; i < ITER; ++i)
			{
				peers[0].send("127.0.0.1", PORT, network::i_p2p_node_with_header::ipv4, 0, data.data(), data.size(),
				              [](network::i_p2p_node_with_header::send_packet_status status, network::header::COMMAND_TYPE command, const char* data, size_t size){
					              if (status != network::i_p2p_node_with_header::send_packet_success)
					              {
						              std::cout << "error: " << status << "\t" << network::i_p2p_node_with_header::send_packet_status_message[status] << std::endl;
					              }
				              });
			}
		}
		
		std::this_thread::sleep_for(std::chrono::seconds(1));
		std::cout << "receive_count is " << receive_count << ", should be " << ITER << std::endl;
		BOOST_CHECK(ITER == receive_count);
		
		peers[0].stop_service();
		peers[1].stop_service();
	}

BOOST_AUTO_TEST_SUITE_END()

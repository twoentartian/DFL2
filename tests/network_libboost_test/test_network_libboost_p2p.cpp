#include <iostream>
#include <filesystem>

#include <glog/logging.h>

#include <network.hpp>

int main(int argc, char **argv)
{
	constexpr char LOG_PATH[] = "./log/";
	//log file path
	{
		google::InitGoogleLogging(argv[0]);
		std::filesystem::path log_path(LOG_PATH);
		if (!std::filesystem::exists(log_path)) std::filesystem::create_directories(log_path);
		google::SetLogDestination(google::INFO, log_path.c_str());
		google::SetStderrLogging(google::INFO);
	}
	
	constexpr int PORT = 1600;
	network::p2p peers[2];
	
	std::atomic_int receive_count = 0;
	peers[1].set_receive_callback([&receive_count](const char* data, size_t length, std::string ip) -> std::string {
		receive_count++;
		return data;
	});
	auto status = peers[1].start_service(PORT);
	LOG_IF(FATAL, status != network::i_p2p_node::success) << "open server failed";
	
	std::string data = "01234567890123456789";

//test connect success
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
	//BOOST_CHECK(ITER == receive_count);
	
	peers[0].stop_service();
	peers[1].stop_service();
}

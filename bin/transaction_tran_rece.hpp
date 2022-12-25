#pragma once

#include <unordered_set>

#include <network.hpp>
#include <mutex>
#include <glog/logging.h>

#include <boost_serialization_wrapper.hpp>

#include <time_util.hpp>

#include "transaction.hpp"
#include "global_types.hpp"
#include "command_allocation.hpp"
#include "std_output.hpp"
#include "block.hpp"
#include "transaction_storage_for_block.hpp"
#include "introducer_data.hpp"
#include "dfl_util.hpp"
#include "env.hpp"

class transaction_tran_rece
{
public:
	using receive_transaction_callback = std::function<void(const transaction& /*trans*/)>;
	using receive_block_callback = std::function<void(const block& /*block*/)>;
	using receive_block_confirmation_callback = std::function<void(const block_confirmation& /*confirmation*/)>;
	
	transaction_tran_rece(const crypto::hex_data& publicKey, const crypto::hex_data& privateKey, const crypto::hex_data& address, std::shared_ptr<transaction_storage_for_block> main_transaction_storage_for_block, bool use_preferred_peer_only = false)
	{
		_public_key = publicKey;
		_private_key = privateKey;
		_address = address;
		
		_main_transaction_storage_for_block = main_transaction_storage_for_block;
		_use_preferred_peer_only = use_preferred_peer_only;
		_running = true;
		_listening = false;
	}
	
	////we need the transaction_storage_for_block to verify/generate block confirmation
	transaction_tran_rece(const std::string& publicKey, const std::string& privateKey, const std::string& address, std::shared_ptr<transaction_storage_for_block> main_transaction_storage_for_block, bool use_preferred_peer_only = false)
	{
		_public_key.assign(publicKey);
		_private_key.assign(privateKey);
		_address.assign(address);
		
		_main_transaction_storage_for_block = main_transaction_storage_for_block;
		_use_preferred_peer_only = use_preferred_peer_only;
		_running = true;
		_listening = false;
	}

	void start_listen(uint16_t listen_port)
	{
		using namespace network;
		_p2p.start_service(listen_port);
		_p2p.set_receive_callback([this](header::COMMAND_TYPE command, const char *data, int length, std::string ip) -> std::tuple<header::COMMAND_TYPE, std::string> {
			if (command == command::transaction)
			{
				//this is a transaction
				std::stringstream ss;
				ss << std::string(data, length);
				transaction trans;
				try
				{
					trans = deserialize_wrap<boost::archive::binary_iarchive, transaction>(ss);
				}
				catch (...)
				{
					LOG(WARNING) << "cannot parse packet data";
					return {command::acknowledge_but_not_accepted, "cannot parse packet data"};
				}
				
				//update _active_peer
				auto peer_iter = _active_peers.find(trans.content.creator.node_address);
				if (peer_iter != _active_peers.end())
				{
					peer_iter->second = time_util::get_current_utc_time();
				}
				
				std::thread temp_thread([this, trans](){
					for (auto&& cb : _receive_transaction_callbacks)
					{
						cb(trans);
					}
				});
				temp_thread.detach();
				
				return {command::acknowledge, ""};
			}
			else if(command == command::block)
			{
				//this is a block
				std::stringstream ss;
				ss << std::string(data, length);
				block blk;
				try
				{
					blk = deserialize_wrap<boost::archive::binary_iarchive, block>(ss);
				}
				catch (...)
				{
					LOG(WARNING) << "cannot parse packet data";
					return {command::acknowledge_but_not_accepted, "cannot parse packet data"};
				}
				
				//find the verified transaction
				std::vector<block_confirmation> confirmations;
				for (auto& [single_transaction_hash, single_transaction]: blk.content.transaction_container)
				{
					if (single_transaction_hash != single_transaction.hash_sha256)
					{
						LOG(WARNING) << "receive a bad block: " << blk.block_content_hash << " from " << blk.content.creator.node_address << ", bad transaction hash";
						return {command::acknowledge_but_not_accepted, "bad block"};
					}
					
					for (auto& [receipt_hash, receipt]: single_transaction.receipts)
					{
						if (receipt_hash != receipt.hash_sha256)
						{
							LOG(WARNING) << "receive a bad block: " << blk.block_content_hash << " from " << blk.content.creator.node_address << ", bad receipt hash";
							return {command::acknowledge_but_not_accepted, "bad block"};
						}
						if (receipt.content.creator.node_address == _address.getTextStr_lowercase())
						{
							//find a receipt generated by me
							if (_main_transaction_storage_for_block->check_verified_transaction(single_transaction) == transaction_storage_for_block::check_receipt_return::pass)
							{
								//generate confirmation
								block_confirmation single_confirmation;
								single_confirmation.creator.node_address = _address.getTextStr_lowercase();
								single_confirmation.creator.node_pubkey = _public_key.getTextStr_lowercase();
								single_confirmation.block_hash = blk.block_content_hash;
								single_confirmation.transaction_hash = single_transaction_hash;
								single_confirmation.receipt_hash = receipt_hash;
								
								auto hash_hex = crypto::sha256_digest(single_confirmation);
								single_confirmation.final_hash = hash_hex.getTextStr_lowercase();
								auto signature = crypto::ecdsa_openssl::sign(hash_hex, _private_key);
								single_confirmation.signature = signature.getTextStr_lowercase();
								
								confirmations.push_back(single_confirmation);
							}
							else
							{
								//do nothing, because the verified transaction is not found in the database
								//TODO: record this abnormal behaviour in log?
							}
						}
					}
				}
				
				//callbacks
				std::thread temp_thread([this, blk](){
					for (auto&& cb : _receive_block_callbacks)
					{
						cb(blk);
					}
				});
				temp_thread.detach();
				
				if (confirmations.empty())
				{
					return {command::acknowledge, ""};
				}
				else
				{
					std::string confirmation_data = serialize_wrap<boost::archive::binary_oarchive>(confirmations).str();
					return {command::block_confirmation, confirmation_data};
				}
			}
			else if (command == command::register_as_peer)
			{
				std::stringstream ss;
				ss << std::string(data, length);
				register_as_peer_data register_request;
				try
				{
					register_request = deserialize_wrap<boost::archive::binary_iarchive, register_as_peer_data>(ss);
				}
				catch (...)
				{
					LOG(WARNING) << "cannot parse packet data";
					return {command::acknowledge_but_not_accepted, "cannot parse packet data"};
				}
				
				//verify signature
				if (!dfl_util::verify_signature(register_request.node_pubkey, register_request.signature, register_request.hash))
				{
					LOG(WARNING) << "verify signature failed";
					return {command::acknowledge_but_not_accepted, "cannot verify signature"};
				}
				
				//verify hash
				if (!dfl_util::verify_hash(register_request, register_request.hash))
				{
					LOG(WARNING) << "verify hash failed";
					return {command::acknowledge_but_not_accepted, "cannot verify hash"};
				}
				
				//add to peer list
				auto [state, msg] = add_peer(register_request.address, register_request.node_pubkey, ip, register_request.port);
				if (!state)
				{
					LOG(WARNING) << "cannot add peer, message:" << msg;
					return {command::acknowledge_but_not_accepted, "cannot add peer, message:" + msg};
				}
				
				return {command::acknowledge, ""};
			}
			
			else
			{
				LOG(WARNING) << "[p2p] unknown command";
				return {command::unknown, ""};
			}
			
		});
		_listening = true;
	}
	
	void broadcast_transaction(const transaction& trans)
	{
		const std::string& trans_hash = trans.hash_sha256;
		
		std::string trans_binary_str = serialize_wrap<boost::archive::binary_oarchive>(trans).str();
		std::unordered_map<std::string, peer_endpoint> peers_copy;
		{
			std::lock_guard guard(_peers_lock);
			peers_copy = _peers;
		}
		
		for (auto&& peer: peers_copy)
		{
			//is this node an introducer?
			if (peer.second.type == peer_endpoint::peer_type::peer_type_introducer) continue;
			
			using namespace network;
			_p2p.send(peer.second.address, peer.second.port, i_p2p_node_with_header::ipv4, command::transaction, trans_binary_str.data(), trans_binary_str.length(), [trans_hash, peer](i_p2p_node_with_header::send_packet_status status, header::COMMAND_TYPE received_command, const char* data, int length){
				std::stringstream ss;
				ss << "[transaction trans] send transaction with hash " << trans_hash << " to " << peer.second.to_string() << ", send status: " << i_p2p_node_with_header::send_packet_status_message[status];
				dfl_util::print_info_to_log_stdcout(ss);
			});
		}
	}
	
	std::vector<block_confirmation> broadcast_block_and_receive_confirmation(const block& blk)
	{
		std::vector<block_confirmation> output_confirmations;
		
		std::string block_binary_str = serialize_wrap<boost::archive::binary_oarchive>(blk).str();
		std::unordered_map<std::string, peer_endpoint> peers_copy;
		{
			std::lock_guard guard(_peers_lock);
			peers_copy = _peers;
		}
		
		for (auto&& peer: peers_copy)
		{
			using namespace network;
			
			//skip not normal peer.
			if (peer.second.type != peer_endpoint::peer_type_normal_node) continue;
			
			_p2p.send(peer.second.address, peer.second.port, i_p2p_node_with_header::ipv4, command::block, block_binary_str.data(), block_binary_str.length(), [&peer, &blk, &output_confirmations](i_p2p_node_with_header::send_packet_status status, header::COMMAND_TYPE command_received, const char* data, int length){
				std::string received_data(data, length);
				if (command_received == command::acknowledge)
				{
					std::stringstream ss;
					ss << "[transaction trans] send block " << blk.block_content_hash << " to " << peer.second.to_string();
					dfl_util::print_info_to_log_stdcout(ss);
					//no confirmation provide, do nothing
				}
				else if (command_received == command::block_confirmation)
				{
					//TODO:check confirmation
					std::vector<block_confirmation> confirmations;
					try
					{
						confirmations = deserialize_wrap<boost::archive::binary_iarchive, std::vector<block_confirmation>>(received_data);
					}
					catch (...)
					{
						LOG(WARNING) << "[transaction trans] error in parsing block confirmation";
						return;
					}
					
					for (auto& confirmation: confirmations)
					{
						output_confirmations.push_back(confirmation);
					}
					
					std::stringstream ss;
					ss << "[transaction trans] send block " << blk.block_content_hash << " to " << peer.second.to_string() << " and receive " << confirmations.size() << " block confirmation";
					dfl_util::print_info_to_log_stdcout(ss);
				}
			});
		}
		
		return output_confirmations;
	}
	
	void set_receive_transaction_callback(receive_transaction_callback callback)
	{
		_receive_transaction_callbacks.push_back(callback);
	}
	
	void set_receive_block_callback(receive_block_callback callback)
	{
		_receive_block_callbacks.push_back(callback);
	}
	
	void set_receive_block_confirmation_callback(receive_block_confirmation_callback callback)
	{
		_receive_block_confirmation_callbacks.push_back(callback);
	}
	
	//name: hash{public key}, address: network address.
	std::tuple<bool, std::string> add_peer(const std::string& name, const std::string& public_key, const std::string& address, uint16_t port)
	{
		if (!dfl_util::verify_address_public_key(name, public_key)) return {false, "wrong public key and name (address) pair"};
		
		peer_endpoint temp(name, public_key, address, port, peer_endpoint::peer_type_normal_node);
		{
			std::lock_guard guard(_peers_lock);
			_peers.emplace(name, temp);
		}
		
		return {true, ""};
	}
	
	std::tuple<bool, std::string> add_preferred_peer(const std::string& name)
	{
		if (_preferred_peers.contains(name))
		{
			return {false, "peer name already exist"};
		}
		else
		{
			_preferred_peers.insert(name);
			return {true, ""};
		}
	}
	
	std::tuple<bool, std::string> add_introducer(const std::string& name, const std::string& address, const std::string& public_key, uint16_t port)
	{
		peer_endpoint temp(name, public_key, address, port, peer_endpoint::peer_type_introducer);
		{
			std::lock_guard guard(_peers_lock);
			_peers.emplace(name, temp);
		}
		
		if (!_registerAndKeeperThread)
		{
			_registerAndKeeperThread.reset(new std::thread([this](){
				while (_running)
				{
					if (!_listening) continue; //server is not allocated a port yet
					
					std::unordered_map<std::string, peer_endpoint> peers_copy;
					{
						std::lock_guard guard(_peers_lock);
						peers_copy = _peers;
					}
					
					//send register request
					{
						register_as_peer_data message;
						message.node_pubkey = _public_key.getTextStr_lowercase();
						message.address = _address.getTextStr_lowercase();
						message.port = _p2p.read_port();
						auto hash_hex = crypto::sha256_digest(message);
						message.hash = hash_hex.getTextStr_lowercase();
						message.signature = crypto::ecdsa_openssl::sign(hash_hex, _private_key).getTextStr_lowercase();
						
						std::string message_str = serialize_wrap<boost::archive::binary_oarchive>(message).str();
						
						using network::i_p2p_node_with_header;
						for (auto& [name, peer] : peers_copy)
						{
							if (peer.type != peer_endpoint::peer_type_introducer) continue;
							
							_p2p.send(peer.address, peer.port, i_p2p_node_with_header::ipv4, command::register_as_peer, message_str.data(), message_str.length(), [this, name](i_p2p_node_with_header::send_packet_status status, network::header::COMMAND_TYPE command_received, const char* data, int length) {
								if (status != i_p2p_node_with_header::send_packet_success)
								{
									LOG(WARNING) << "[transaction trans] register as peer on " << name << " does not success, status: " << i_p2p_node_with_header::send_packet_status_message[status];
									return;
								}
								
								if (command_received == command::acknowledge)
								{
                                    std::string msg(data, length);
                                    if (msg == DFL_MESSAGE::PEER_REGISTER_ALREADY_EXIST)
                                    {
                                        //do nothing because we know they have put us on the peer list
                                        return;
                                    }
                                    if (msg == DFL_MESSAGE::PEER_REGISTER_NEW_PEER)
                                    {
                                        LOG(INFO) << "[transaction trans] register as peer on " << name << " successfully";
                                        return;
                                    }
                                    
								}
								else
								{
									LOG(WARNING) << "[transaction trans] WARNING, register as peer on " << name << " but does not return acknowledge, return command: " << command_received << "-" << std::string(data, length);
									return;
								}
							});
						}
					}
					
					//update the active peer list
					{
						std::vector<std::string> removal_list;
						time_t now = time_util::get_current_utc_time();
						for (auto& [peer_name, last_active_time]: _active_peers)
						{
							if (now - last_active_time > _inactive_time_seconds)
							{
								removal_list.push_back(peer_name);
							}
						}
						for (auto& removal_peer : removal_list)
						{
							_active_peers.erase(removal_peer);
						}
					}
					
					bool exit = false;
					for (int i = 0; i < 10; ++i)
					{
						std::this_thread::sleep_for(std::chrono::milliseconds(500));
						if (!_running)
						{
							exit = true;
							break;
						}
					}
					if (exit)
					{
						break;
					}
					
				}
			}));
		}
		
		return {true, ""};
	}
	
	//remove certain amount of peers, the address on the remove list is preferred to be removed.
	std::vector<std::tuple<std::string, peer_endpoint>> remove_peers(int remove_count, std::vector<std::string> remove_list = {})
	{
		std::vector<std::tuple<std::string, peer_endpoint>> output;
		int remove_list_index = 0;
		while (remove_count > 0)
		{
			auto index_to_remove = remove_list[remove_list_index];
			if (_peers.find(index_to_remove) != _peers.end())
			{
				auto remove_item = std::make_tuple(index_to_remove, _peers[index_to_remove]);
				if (_peers.erase(index_to_remove) == 1)
				{
					output.push_back(remove_item);
					remove_count--;
				}
			}
			else
			{
				if (_peers.size() == 0)
					return output;
				index_to_remove = _peers.size() - 1;
				auto remove_item = std::make_tuple(index_to_remove, _peers[index_to_remove]);
				if (_peers.erase(index_to_remove) == 1)
				{
					output.push_back(remove_item);
					remove_count--;
				}
			}
			remove_list_index++;
		}
		
		return output;
	}
	
	std::tuple<bool, std::string> try_to_add_peer(int desired_peer_count = -1)
	{
		if (desired_peer_count == -1) desired_peer_count = _maximum_peer;
		if (desired_peer_count < 0) return {false, "invalid peer count"};
		
		std::unordered_map<std::string, peer_endpoint> peers_copy;
		{
			std::lock_guard guard(_peers_lock);
			peers_copy = _peers;
		}
		
		//build request peer message
		request_peer_info_data message;
		message.requester_address = _address.getTextStr_lowercase();
		message.peers_info = {};
		message.generator_address = _address.getTextStr_lowercase();
		message.node_pubkey = _public_key.getTextStr_lowercase();
		
		//hash and signature
		byte_buffer buffer;
		message.to_byte_buffer(buffer);
		crypto::hex_data hash = crypto::sha256::digest_s(buffer.data(), buffer.size());
		message.hash = hash.getTextStr_lowercase();
		crypto::hex_data sig = crypto::ecdsa_openssl::sign(hash, _private_key);
		message.signature = sig.getTextStr_lowercase();
		
		//serialize
		std::string message_str = serialize_wrap<boost::archive::binary_oarchive>(message).str();
		
		for (auto&& peer: peers_copy)
		{
			if (peer.second.type != peer_endpoint::peer_type_introducer) continue;
			using namespace network;
			
			//send available peer list request
			_p2p.send(peer.second.address, peer.second.port, i_p2p_node_with_header::ipv4, command::request_peer_info, message_str.data(), message_str.length(), [this](i_p2p_node_with_header::send_packet_status status, header::COMMAND_TYPE command_received, const char* data, int length) {
				if (status != i_p2p_node_with_header::send_packet_success)
				{
					LOG(WARNING) << "[transaction trans] request peer list does not success, status:" << i_p2p_node_with_header::send_packet_status_message[status];
					return;
				}
				
				if (command_received == command::reply_peer_info)
				{
					std::string received_data(data, length);
					request_peer_info_data received_peers;
					try
					{
						received_peers = deserialize_wrap<boost::archive::binary_iarchive, request_peer_info_data>(received_data);
					}
					catch (...)
					{
						LOG(WARNING) << "[transaction trans] WARNING, error in parsing the peers information";
						return;
					}
					
					//check name
					if (crypto::hex_data(received_peers.requester_address) != _address)
					{
						LOG(WARNING) << "[transaction trans] WARNING, the returned peers information has a wrong address";
						return;
					}
					
					//register on the other nodes as peer
					register_as_peer_data message;
					message.node_pubkey = _public_key.getTextStr_lowercase();
					message.address = _address.getTextStr_lowercase();
					message.port = _p2p.read_port();
					auto hash_hex = crypto::sha256_digest(message);
					message.hash = hash_hex.getTextStr_lowercase();
					message.signature = crypto::ecdsa_openssl::sign(hash_hex, _private_key).getTextStr_lowercase();
					
					std::string message_str = serialize_wrap<boost::archive::binary_oarchive>(message).str();
					for (auto& single_peer_info : received_peers.peers_info)
					{
						crypto::hex_data single_peer_name_hex(single_peer_info.name);
						if (single_peer_name_hex == _address) // myself
						{
							continue;
						}
						
						//send register as peer request
						_p2p.send(single_peer_info.address, single_peer_info.port, i_p2p_node_with_header::ipv4, command::register_as_peer, message_str.data(), message_str.length(), [this, single_peer_info](i_p2p_node_with_header::send_packet_status status, header::COMMAND_TYPE command_received, const char* data, int length) {
							if (status != i_p2p_node_with_header::send_packet_success)
							{
								LOG(WARNING) << "[transaction trans] register as peer does not success, status: " << i_p2p_node_with_header::send_packet_status_message[status];
								return;
							}
							
							if (command_received == command::acknowledge)
							{
								//add the peer to the active list
								_active_peers[single_peer_info.name] = time_util::get_current_utc_time();
							}
							else
							{
								LOG(WARNING) << "[transaction trans] WARNING, register as peer but does not return acknowledge, return command: " << command_received << "-" << std::string(data, length);
								return;
							}
						});
					}
				}
				else
				{
					LOG(WARNING) << "[transaction trans] WARNING, request peer info but returned a wrong command: " << command_received;
				}
			});
		}
		
		return {true, ""};
	}
	
	[[nodiscard]] std::unordered_map<std::string, peer_endpoint> get_peers()
	{
		return _peers;
	}
	
	void stop_listening()
	{
		_p2p.stop_service();
		_listening = false;
	}
	
	~transaction_tran_rece()
	{
		stop_listening();
		_running = false;
		if (_registerAndKeeperThread)
		{
			_registerAndKeeperThread->join();
		}
	}
	
	GENERATE_GET(_maximum_peer, get_maximum_peer);
	GENERATE_GET(_use_preferred_peer_only, get_use_preferred_peer_only);
	GENERATE_GET(_inactive_time_seconds, get_inactive_time);
private:
	crypto::hex_data _address;
	crypto::hex_data _public_key;
	crypto::hex_data _private_key;
	
	size_t _maximum_peer;
	bool _use_preferred_peer_only;
	std::unordered_set<std::string> _preferred_peers;
	
	std::vector<receive_transaction_callback> _receive_transaction_callbacks;
	std::vector<receive_block_callback> _receive_block_callbacks;
	std::vector<receive_block_confirmation_callback> _receive_block_confirmation_callbacks;
	network::p2p_with_header _p2p;
	std::unordered_map<std::string, peer_endpoint> _peers;
	std::mutex _peers_lock;
	std::shared_ptr<transaction_storage_for_block> _main_transaction_storage_for_block;
	
	std::unordered_map<std::string, uint64_t> _active_peers;
	uint64_t _inactive_time_seconds;
	
	std::shared_ptr<std::thread> _registerAndKeeperThread;
	bool _running;
	bool _listening;
};

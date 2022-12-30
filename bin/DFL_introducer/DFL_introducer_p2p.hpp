#pragma once

#include <unordered_map>
#include <functional>
#include <utility>
#include <vector>
#include <random>
#include <chrono>

#include <network.hpp>
#include <boost_serialization_wrapper.hpp>

#include "../command_allocation.hpp"
#include "../global_types.hpp"
#include "../dfl_util.hpp"
#include "../env.hpp"

class introducer_p2p
{
public:
    using new_peer_callback = std::function<void(const peer_endpoint & /*peer*/)>;
    using peer_expire_callback = std::function<void(const peer_endpoint & /*peer*/)>;
    
    introducer_p2p(const std::string &address, const std::string &privateKey, const std::string &publicKey)
    {
        _address.assign(address);
        _private_key.assign(privateKey);
        _public_key.assign(publicKey);
        _running = false;
        expire_second = 60;
    }
    
    introducer_p2p(const crypto::hex_data &address, const crypto::hex_data &privateKey,
                   const crypto::hex_data &publicKey)
    {
        _address = address;
        _private_key = privateKey;
        _public_key = publicKey;
        _running = false;
        expire_second = 60;
    }
    
    void set_expire_second(size_t arg_expire_second)
    {
        expire_second = arg_expire_second;
    }
    
    ~introducer_p2p()
    {
        stop_listen();
    }
    
    struct peer_endpoint_info
    {
        peer_endpoint peer_ep;
        std::chrono::time_point<std::chrono::system_clock> last_seen_time;
        
        peer_endpoint_info(peer_endpoint arg_peer_ep, std::chrono::time_point<std::chrono::system_clock> arg_last_seen_time) : peer_ep(std::move(arg_peer_ep)), last_seen_time(arg_last_seen_time)
        {
        
        }
    };
    
    void start_listen(uint16_t listen_port)
    {
        using namespace network;
        _p2p.set_receive_callback([this](header::COMMAND_TYPE command, const char *data, int length, const std::string &ip) -> std::tuple<header::COMMAND_TYPE, std::string>
                                  {
                                      if (command == command::register_as_peer)
                                      {
                                          //this is a register request
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
                                              return {command::acknowledge_but_not_accepted,"cannot parse packet data"};
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
                
                                          //verify node public key and address
                                          if (!dfl_util::verify_address_public_key(register_request.address,register_request.node_pubkey))
                                          {
                                              LOG(WARNING) << "verify node public key failed";
                                              return {command::acknowledge_but_not_accepted,"cannot verify node public key / address"};
                                          }
    
                                          //add to peer list
                                          auto [state, msg] = add_peer(register_request.address, register_request.node_pubkey, ip, register_request.port);
                                          if (!state)
                                          {
                                              LOG(WARNING) << "cannot add peer: " << msg;
                                              return {command::acknowledge_but_not_accepted, "cannot add peer: " + msg};
                                          }
                                          if (msg == DFL_MESSAGE::PEER_REGISTER_ALREADY_EXIST)
                                          {
                                              return {command::acknowledge_but_not_effective, msg};
                                          }
                                          return {command::acknowledge, msg};
                                      }
                                      else if (command == command::request_peer_info)
                                      {
                                          //request for peer info
                                          std::stringstream ss;
                                          ss << std::string(data, length);
                                          request_peer_info_data peer_request;
                                          try
                                          {
                                              peer_request = deserialize_wrap<boost::archive::binary_iarchive, request_peer_info_data>(ss);
                                          }
                                          catch (...)
                                          {
                                              LOG(WARNING) << "cannot parse packet data";
                                              return {command::acknowledge_but_not_accepted,"cannot parse packet data"};
                                          }
                
                                          if (peer_request.requester_address != peer_request.generator_address)
                                          {
                                              return {command::acknowledge_but_not_accepted,"invalid requester and generator address"};
                                          }
                
                                          //verify signature
                                          if (!dfl_util::verify_signature(peer_request.node_pubkey, peer_request.signature, peer_request.hash))
                                          {
                                              LOG(WARNING) << "verify signature failed";
                                              return {command::acknowledge_but_not_accepted, "cannot verify signature"};
                                          }
                
                                          //verify hash
                                          if (!dfl_util::verify_hash(peer_request, peer_request.hash))
                                          {
                                              LOG(WARNING) << "verify hash failed";
                                              return {command::acknowledge_but_not_accepted, "cannot verify hash"};
                                          }
                
                                          //verify node public key and address
                                          if (!dfl_util::verify_address_public_key(peer_request.generator_address,peer_request.node_pubkey))
                                          {
                                              LOG(WARNING) << "verify node public key failed";
                                              return {command::acknowledge_but_not_accepted,"cannot verify node public key / address"};
                                          }
                
                                          //add peer
                                          decltype(this->_peers) peers_copy;
                                          {
                                              std::lock_guard guard(_peers_lock);
                                              peers_copy = _peers;
                                          }
                                          for (auto &[peer_name, peer]: peers_copy)
                                          {
                                              peer_request.peers_info.push_back(peer.peer_ep);
                                          }
                
                                          //shuffle peer list
                                          {
                                              std::random_device rd;
                                              std::mt19937 g(rd());
                                              std::shuffle(peer_request.peers_info.begin(),peer_request.peers_info.end(), g);
                                          }
                
                                          peer_request.generator_address = _address.getTextStr_lowercase();
                                          peer_request.node_pubkey = _public_key.getTextStr_lowercase();
                                          peer_request.hash = crypto::sha256_digest(peer_request).getTextStr_lowercase();
                                          auto sig_hex = crypto::ecdsa_openssl::sign(peer_request.hash, _private_key);
                                          peer_request.signature = sig_hex.getTextStr_lowercase();
                
                                          std::string reply = serialize_wrap<boost::archive::binary_oarchive>(peer_request).str();
                
                                          return {command::reply_peer_info, reply};
                                      }
                                      else
                                      {
                                          LOG(WARNING) << "[p2p] unknown command";
                                          return {command::unknown, ""};
                                      }
                                  });
        _p2p.start_service(listen_port);
        
        _running = true;
        if (!_check_peer_thread)
        {
            _check_peer_thread.reset(new std::thread([this]()
                                                     {
                                                         this->check_expire_peer_func();
                                                     }));
        }
    }
    
    void stop_listen()
    {
        _running = false;
        if (_check_peer_thread)
        {
            _check_peer_thread->join();
        }
    }
    
    //name: hash{public key}, address: network address.
    std::tuple<bool, std::string> add_peer(const std::string &name, const std::string &public_key, const std::string &address, uint16_t port)
    {
        if (!dfl_util::verify_address_public_key(name, public_key)) return {false, "wrong public key and name (address) pair"};
        
        auto iter = _peers.find(name);
        if (iter != _peers.end())
        {
            bool skip = true;
            auto peer = iter->second.peer_ep;
            if (public_key != peer.public_key) skip = false;
            if (address != peer.address) skip = false;
            if (port != peer.port) skip = false;
            if (skip)
            {
                //peer is already registered
                iter->second.last_seen_time = std::chrono::system_clock::now();
                return {true, DFL_MESSAGE::PEER_REGISTER_ALREADY_EXIST};
            }
        }
        
        peer_endpoint temp(name, public_key, address, port, peer_endpoint::peer_type_normal_node);
        peer_endpoint_info temp_info(temp, std::chrono::system_clock::now());
        {
            std::lock_guard guard(_peers_lock);
            _peers.emplace(name, temp_info);
        }
        
        for (auto &cb: _new_peer_callbacks)
        {
            cb(temp);
        }
        
        return {true, DFL_MESSAGE::PEER_REGISTER_NEW_PEER};
    }
    
    void add_new_peer_callback(const new_peer_callback &cb)
    {
        _new_peer_callbacks.push_back(cb);
    }
    
    void add_peer_expire_callback(const peer_expire_callback &cb)
    {
        _peer_expire_callbacks.push_back(cb);
    }
    
    [[nodiscard]] std::unordered_map<std::string, peer_endpoint_info> get_peers()
    {
        return _peers;
    }

private:
    void check_expire_peer_func()
    {
        while (_running)
        {
            std::vector<std::string> peers_to_remove;
            auto time_now = std::chrono::system_clock::now();
            for (const auto &[name, single_peer]: _peers)
            {
                auto elapsed = time_now - single_peer.last_seen_time;
                if (elapsed > std::chrono::seconds(expire_second))
                    peers_to_remove.push_back(name);
            }
            for (const auto &peer_to_remove: peers_to_remove)
            {
                auto removed_target = _peers.find(peer_to_remove);
                auto peer_ep = removed_target->second.peer_ep;
                _peers.erase(removed_target);
                for (const auto &cb: _peer_expire_callbacks)
                {
                    cb(peer_ep);
                }
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    
    network::p2p_with_header _p2p;
    
    std::unordered_map<std::string, peer_endpoint_info> _peers;
    std::mutex _peers_lock;
    
    size_t expire_second;
    bool _running;
    std::shared_ptr<std::thread> _check_peer_thread;
    
    crypto::hex_data _address;
    crypto::hex_data _public_key;
    crypto::hex_data _private_key;
    
    std::vector<new_peer_callback> _new_peer_callbacks;
    std::vector<peer_expire_callback> _peer_expire_callbacks;
};

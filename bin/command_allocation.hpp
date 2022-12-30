#pragma once

#include <iostream>

class command
{
public:
	static constexpr uint16_t unknown = 0;
	static constexpr uint16_t acknowledge = 1;
    static constexpr uint16_t acknowledge_but_not_effective = 2;
	static constexpr uint16_t acknowledge_but_not_accepted = 3;
	static constexpr uint16_t transaction = 4;
	static constexpr uint16_t block = 5;
	static constexpr uint16_t block_confirmation = 6;
	
	static constexpr uint16_t register_as_peer = 7;
	static constexpr uint16_t request_peer_info = 8;
	static constexpr uint16_t reply_peer_info = 9;
	
	
};
#pragma once

#include "./hash.hpp"
#include "./hex_data.hpp"
#include "./crypto_config.hpp"
#if USE_OPENSSL
#include <openssl/evp.h>
#include <openssl/sha.h>
#endif

#define SHA2_SHFR(x, n)	(x >> n)
#define SHA2_ROTR(x, n)   ((x >> n) | (x << ((sizeof(x) << 3) - n)))
#define SHA2_CH(x, y, z)  ((x & y) ^ (~x & z))
#define SHA2_MAJ(x, y, z) ((x & y) ^ (x & z) ^ (y & z))
#define SHA256_F1(x) (SHA2_ROTR(x,  2) ^ SHA2_ROTR(x, 13) ^ SHA2_ROTR(x, 22))
#define SHA256_F2(x) (SHA2_ROTR(x,  6) ^ SHA2_ROTR(x, 11) ^ SHA2_ROTR(x, 25))
#define SHA256_F3(x) (SHA2_ROTR(x,  7) ^ SHA2_ROTR(x, 18) ^ SHA2_SHFR(x,  3))
#define SHA256_F4(x) (SHA2_ROTR(x, 17) ^ SHA2_ROTR(x, 19) ^ SHA2_SHFR(x, 10))
#define SHA2_UNPACK32(x, str)				 \
{											  \
	*((str) + 3) = (uint8_t) ((x)	  );	   \
	*((str) + 2) = (uint8_t) ((x) >>  8);	   \
	*((str) + 1) = (uint8_t) ((x) >> 16);	   \
	*((str) + 0) = (uint8_t) ((x) >> 24);	   \
}
#define SHA2_PACK32(str, x)	                \
{								            \
	*(x) =   ((uint32_t) *((str) + 3)	  )	\
		   | ((uint32_t) *((str) + 2) <<  8)	\
		   | ((uint32_t) *((str) + 1) << 16)	\
		   | ((uint32_t) *((str) + 0) << 24); \
}
namespace crypto
{
	class sha256 : hash {
    protected:
        const uint32_t sha256_k[64] =
                {0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
                 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
                 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
                 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
                 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
                 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
                 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
                 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
                 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
                 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
                 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
                 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
                 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
                 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
                 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
                 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};
        static const uint32_t SHA256_BLOCK_SIZE = (512 / 8);
        uint32_t m_tot_len;
        uint32_t m_len;
        uint8_t m_block[SHA256_BLOCK_SIZE];
        uint32_t m_h[8];
	
	private:
		void transform(const uint8_t*message,uint32_t block_nb)
		{
			uint32_t w[64];
			uint32_t wv[8];
			uint32_t t1, t2;
			const uint8_t *sub_block;
			int i;
			int j;
			for (i = 0; i < (int) block_nb; i++) {
				sub_block = message + (i << 6);
				for (j = 0; j < 16; j++) {
					SHA2_PACK32(&sub_block[j << 2], &w[j]);
				}
				for (j = 16; j < 64; j++) {
					w[j] =  SHA256_F4(w[j -  2]) + w[j -  7] + SHA256_F3(w[j - 15]) + w[j - 16];
				}
				for (j = 0; j < 8; j++) {
					wv[j] = m_h[j];
				}
				for (j = 0; j < 64; j++) {
					t1 = wv[7] + SHA256_F2(wv[4]) + SHA2_CH(wv[4], wv[5], wv[6])
					     + sha256_k[j] + w[j];
					t2 = SHA256_F1(wv[0]) + SHA2_MAJ(wv[0], wv[1], wv[2]);
					wv[7] = wv[6];
					wv[6] = wv[5];
					wv[5] = wv[4];
					wv[4] = wv[3] + t1;
					wv[3] = wv[2];
					wv[2] = wv[1];
					wv[1] = wv[0];
					wv[0] = t1 + t2;
				}
				for (j = 0; j < 8; j++) {
					m_h[j] += wv[j];
				}
			}
		}
		
		void init()
		{
			m_h[0] = 0x6a09e667;
			m_h[1] = 0xbb67ae85;
			m_h[2] = 0x3c6ef372;
			m_h[3] = 0xa54ff53a;
			m_h[4] = 0x510e527f;
			m_h[5] = 0x9b05688c;
			m_h[6] = 0x1f83d9ab;
			m_h[7] = 0x5be0cd19;
			m_len = 0;
			m_tot_len = 0;
		}
		
		void update(const uint8_t *message, uint32_t length)
		{
			uint32_t block_nb;
			uint32_t new_len,rem_len,tmp_len;
			const uint8_t *shifted_message;
			tmp_len = SHA256_BLOCK_SIZE - m_len;
			rem_len = length< tmp_len ? length : tmp_len;
			memcpy(&m_block[m_len], message, rem_len);
			if (m_len + length < SHA256_BLOCK_SIZE) {
				m_len += length;
				return;
			}
			new_len = length - rem_len;
			block_nb = new_len / SHA256_BLOCK_SIZE;
			shifted_message = message + rem_len;
			transform(m_block, 1);
			transform(shifted_message, block_nb);
			rem_len = new_len % SHA256_BLOCK_SIZE;
			memcpy(m_block, &shifted_message[block_nb << 6], rem_len);
			m_len = rem_len;
			m_tot_len += (block_nb + 1) << 6;
			
		}
		
		void final(uint8_t *digest)
		{
			uint32_t block_nb;
			uint32_t pm_len;
			uint32_t len_b;
			int i;
			block_nb = (1 + ((SHA256_BLOCK_SIZE - 9) < (m_len % SHA256_BLOCK_SIZE)));
			len_b = (m_tot_len + m_len) << 3;
			pm_len = block_nb << 6;
			memset(m_block + m_len, 0, pm_len - m_len);
			m_block[m_len] = 0x80;
			SHA2_UNPACK32(len_b, m_block + pm_len - 4);
			transform(m_block, block_nb);
			for (i = 0 ; i < 8; i++) {
				SHA2_UNPACK32(m_h[i], &digest[i << 2]);
			}
		}
    
    public:
        constexpr static int OUTPUT_SIZE = 256;
		constexpr static int DIGEST_SIZE = (OUTPUT_SIZE / 8);//bits
	    
        hex_data digest(const std::string &message) override
        {
	        return digest((uint8_t*)message.data(), message.size());
        }
        
        static hex_data digest_s(const std::string &message)
        {
	        return digest_s((uint8_t*)message.data(), message.size());
        }
		
		hex_data digest(const uint8_t* data, size_t size) override
		{
			uint8_t digest[DIGEST_SIZE];
			memset(digest,0,DIGEST_SIZE);
			init();
			update(data, size);
			final(digest);
			return hex_data(digest, DIGEST_SIZE);
		}
		
		static hex_data digest_s(const uint8_t* data, size_t size)
		{
			sha256 ctx;
			uint8_t digest[DIGEST_SIZE];
			memset(digest,0,DIGEST_SIZE);
			ctx.init();
			ctx.update(data, size);
			ctx.final(digest);
			return hex_data(digest, DIGEST_SIZE);
		}

#if USE_OPENSSL
		hex_data digest_openssl(const std::string& message) override
		{
			return digest_openssl((uint8_t*)message.data(), message.size());
		}
		
		hex_data digest_openssl(const uint8_t* data, size_t size)
		{
			std::string hash;
			hash.resize(DIGEST_SIZE);
			SHA256((const unsigned char *)data, size, (unsigned char *)hash.c_str());
			hex_data output(hash.c_str(), DIGEST_SIZE);
			return output;
		}
		
		static hex_data digest_openssl_s(const std::string& message)
		{
			return digest_openssl_s((uint8_t*)message.data(), message.size());
		}
		
		static hex_data digest_openssl_s(const uint8_t* data, size_t size)
		{
			std::string hash;
			hash.resize(DIGEST_SIZE);
			SHA256((const unsigned char *)data, size, (unsigned char *)hash.c_str());
			hex_data output(hash.c_str(), DIGEST_SIZE);
			return output;
		}
#endif
    };

	template <typename T>
	hex_data sha256_digest(T target)
	{
		byte_buffer buffer;
		target.to_byte_buffer(buffer);
#if USE_OPENSSL
		hex_data output = sha256::digest_openssl_s(buffer.data(), buffer.size());
#else
		hex_data output = sha256::digest_s(buffer.data(), buffer.size());
#endif
		return output;
	}
	
}

#undef SHA2_SHFR
#undef SHA2_ROTR
#undef SHA2_CH
#undef SHA2_MAJ
#undef SHA256_F1
#undef SHA256_F2
#undef SHA256_F3
#undef SHA256_F4
#undef SHA2_UNPACK32
#undef SHA2_PACK32

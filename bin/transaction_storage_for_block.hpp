#pragma once

#if USE_ROCKSDB
#include <rocksdb_api.hpp>
#else
#include <unordered_map>
#endif

#include <auto_multi_thread.hpp>
#include <unordered_map>

#include "./transaction.hpp"

class transaction_storage_for_block
{
public:
	static constexpr char sub_dir_for_block_cache[] = "cache";
	static constexpr char sub_dir_for_verified_transactions[] = "verified";
	
	transaction_storage_for_block(const std::string &db_path)
	{
		_block_cache_size = 0;
		
#if USE_ROCKSDB
		//_db_verified_transactions
		{
			_db_path_sub_verified_transactions.assign(std::filesystem::path(db_path) / sub_dir_for_verified_transactions);
			rocksdb::Options options;
			rocksdb::Status status;
			options.create_if_missing = true;
			//options.IncreaseParallelism();
			options.OptimizeLevelStyleCompaction();
			options.max_total_wal_size = 10 * 1024 * 1024; //10MB, https://github.com/facebook/rocksdb/blob/master/include/rocksdb/options.h#L477
			
			rocksdb::BlockBasedTableOptions table_options;
			table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10));
			table_options.block_cache = rocksdb::NewLRUCache(100 * 1024 * 1024); //https://github.com/EighteenZi/rocksdb_wiki/blob/master/Memory-usage-in-RocksDB.md
			options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_options));
			status = rocksdb::DB::Open(options, _db_path_sub_verified_transactions.string(), &_db_verified_transactions);
			CHECK(status.ok()) << "[transaction_storage_for_block] failed to open rocksdb for _db_verified_transactions";
		}
		
		//_db_block_cache
		{
			_db_path_sub_block_cache.assign(std::filesystem::path(db_path) / sub_dir_for_block_cache);
			rocksdb::Options options;
			rocksdb::Status status;
			options.create_if_missing = true;
			options.IncreaseParallelism();
			options.OptimizeLevelStyleCompaction();
			options.max_total_wal_size = 10 * 1024 * 1024; //10MB, https://github.com/facebook/rocksdb/blob/master/include/rocksdb/options.h#L477
			
			rocksdb::BlockBasedTableOptions table_options;
			table_options.filter_policy.reset(rocksdb::NewBloomFilterPolicy(10));
			table_options.block_cache = rocksdb::NewLRUCache(100 * 1024 * 1024); //https://github.com/EighteenZi/rocksdb_wiki/blob/master/Memory-usage-in-RocksDB.md
			options.table_factory.reset(rocksdb::NewBlockBasedTableFactory(table_options));
			status = rocksdb::DB::Open(options, _db_path_sub_block_cache.string(), &_db_block_cache);
			CHECK(status.ok()) << "[transaction_storage_for_block] failed to open rocksdb for _db_block_cache";
		}
		
		rocksdb::Iterator *it = _db_block_cache->NewIterator(rocksdb::ReadOptions());
		for (it->SeekToFirst(); it->Valid(); it->Next())
		{
			_block_cache_size++;
		}
		LOG_IF(ERROR, !it->status().ok()) << "error to retrieve the block cache database";
		delete it;
#else

#endif

	}
	
	~transaction_storage_for_block()
	{
#if USE_ROCKSDB
		LOG(INFO) << "flush block cache database";
		_db_block_cache->FlushWAL(true);
		
		LOG(INFO) << "flush verified transaction database";
		_db_verified_transactions->FlushWAL(true);
#else
		LOG(INFO) << "block cache database is lost due to BlockDB disabled";
		LOG(INFO) << "verified transaction database is lost due to BlockDB disabled";
#endif
	}

#pragma region Verified transaction database API
	enum class check_receipt_return
	{
		unknown,
		pass,
		not_found,
		mismatch,
		receipt_not_present
	};
	
	static std::string to_string(check_receipt_return t)
    {
        switch (t)
        {
            case check_receipt_return::unknown:
                return "unknown";
            case check_receipt_return::pass:
                return "pass";
            case check_receipt_return::not_found:
                return "not_found";
            case check_receipt_return::mismatch:
                return "mismatch";
            case check_receipt_return::receipt_not_present:
                return "receipt_not_present";
        }
        return "unknown";
    }

	void add_verified_transaction(const transaction& target_transaction, const transaction_receipt& receipt)
	{
		verified_transaction_item item;
		item.hash_sha256 = target_transaction.hash_sha256;
		item.signature = target_transaction.signature;
		item.receipt_hash = receipt.hash_sha256;
		item.receipt = receipt;

#if USE_ROCKSDB
		auto item_str = serialize_wrap<boost::archive::binary_oarchive>(item).str();
		
		std::string db_data;
		auto status = _db_verified_transactions->Get(rocksdb::ReadOptions(), item.hash_sha256, &db_data);
		if (status.ok())
		{
			LOG(WARNING) << "[transaction_storage_for_block] overwrite verified transactions: " << item.hash_sha256 << " with receipt: " << receipt.hash_sha256;
		}
		status = _db_verified_transactions->Put(rocksdb::WriteOptions(), item.hash_sha256, item_str);
		LOG_IF(WARNING, !status.ok()) << "[transaction_storage_for_block] failed to add verified transactions: " << item.hash_sha256;
#else
		auto iter = _map_verified_transactions.find(item.hash_sha256);
		if (iter == _map_verified_transactions.end())
		{
			_map_verified_transactions.emplace(item.hash_sha256, item);
		}
		else
		{
			LOG(WARNING) << "[transaction_storage_for_block] overwrite verified transactions: " << item.hash_sha256 << " with receipt: " << receipt.hash_sha256;
			iter->second = item;
		}
#endif
	}
	
	check_receipt_return check_verified_transaction(const transaction& target_transaction)
	{
#if USE_ROCKSDB
		std::string db_data;
		auto status = _db_verified_transactions->Get(rocksdb::ReadOptions(), target_transaction.hash_sha256, &db_data);
		if (!status.ok())
		{
			return check_receipt_return::not_found;
		}
		verified_transaction_item vti = deserialize_wrap<boost::archive::binary_iarchive, verified_transaction_item>(db_data);
#else
		auto iter = _map_verified_transactions.find(target_transaction.hash_sha256);
		if (iter == _map_verified_transactions.end())
		{
			return check_receipt_return::not_found;
		}
		verified_transaction_item& vti = iter->second;
#endif
		
		//check transaction
		if (vti.signature != target_transaction.signature)
		{
			return check_receipt_return::mismatch;
		}
		auto target_receipt = target_transaction.receipts.find(vti.receipt_hash);
		if (target_receipt == target_transaction.receipts.end())
		{
			return check_receipt_return::receipt_not_present;
		}
		if (target_receipt->second != vti.receipt)
		{
			return check_receipt_return::mismatch;
		}
		
		return check_receipt_return::pass;
	}
	
	bool remove_verified_transaction(const transaction& target_transaction)
	{
#if USE_ROCKSDB
		auto status = _db_verified_transactions->Delete(rocksdb::WriteOptions(), target_transaction.hash_sha256);
		return status.ok();
#else
		auto iter = _map_verified_transactions.find(target_transaction.hash_sha256);
		if (iter == _map_verified_transactions.end())
		{
			return false;
		}
		else
		{
			_map_verified_transactions.erase(iter);
			return true;
		}
#endif
	}
#pragma endregion

#pragma region Block cache API
	void add_to_block_cache(const transaction& target_transaction)
	{
#if USE_ROCKSDB
		std::lock_guard guard(_db_block_cache_lock);
		std::string data_in_db;
		auto status = _db_block_cache->Get(rocksdb::ReadOptions(), target_transaction.hash_sha256, &data_in_db);
		bool already_exist = status.ok();
#else
		auto iter = _map_block_cache.find(target_transaction.hash_sha256);
		bool already_exist = !(iter == _map_block_cache.end());
#endif
		if (already_exist)
		{
			//the transaction is already in the database
#if USE_ROCKSDB
			transaction trans_in_db = deserialize_wrap<boost::archive::binary_iarchive, transaction>(data_in_db);
#else
			transaction trans_in_db = iter->second;
#endif
			bool changed = false;
			for (auto& single_recep: target_transaction.receipts)
			{
				if (trans_in_db.receipts.find(single_recep.first) == trans_in_db.receipts.end())
				{
					//new transaction receipts
					trans_in_db.receipts.emplace(single_recep.first, single_recep.second);
					_receipt_count[target_transaction.hash_sha256]++;
					changed = true;
				}
			}
			
			//write back to database
			if (changed)
			{
#if USE_ROCKSDB
				std::string trans_in_db_str = serialize_wrap<boost::archive::binary_oarchive>(trans_in_db).str();
				_db_block_cache->Put(rocksdb::WriteOptions(), trans_in_db.hash_sha256, trans_in_db_str);
#else
				_map_block_cache[trans_in_db.hash_sha256] = trans_in_db;
#endif
			}
		}
		else
		{
			//add this new transaction to the database
			if (!target_transaction.receipts.empty()) return; //not a new transaction because it has receipts. It might be a late transaction which has been dumped.

#if USE_ROCKSDB
			std::string transaction_data_str = serialize_wrap<boost::archive::binary_oarchive>(target_transaction).str();
			_db_block_cache->Put(rocksdb::WriteOptions(), target_transaction.hash_sha256, transaction_data_str);
#else
			_map_block_cache.emplace(target_transaction.hash_sha256, target_transaction);
#endif
			_block_cache_size++;
			_receipt_count[target_transaction.hash_sha256] = 0;
		}
	}
	
	size_t block_cache_size()
	{
		return _block_cache_size;
	}
	
	/**
	 * transaction with more than or equal to the {dump_threshold} receipts will be dumped. older transactions in the database will also be dumped
	 * @param dump_threshold
	 * @return
	 */
	std::vector<transaction> dump_block_cache(int dump_threshold)
	{
		std::vector<transaction> output;
#if USE_ROCKSDB
		std::vector<std::string> all_transaction_key_data;
		std::vector<std::string> transactions_str;
		{
			std::lock_guard guard(_db_block_cache_lock);
			rocksdb::Iterator* it = _db_block_cache->NewIterator(rocksdb::ReadOptions());
			for (it->SeekToFirst(); it->Valid(); it->Next())
			{
				all_transaction_key_data.push_back(it->key().ToString());
			}
			assert(it->status().ok());
            delete it;
			transactions_str.reserve(all_transaction_key_data.size());
			for (auto& single_transaction_key: all_transaction_key_data)
			{
				bool dump = false;
				auto iter = _receipt_count.find(single_transaction_key);
				if (iter == _receipt_count.end())
				{
					dump = true;
				}
				else
				{
					if (iter->second >= dump_threshold)	dump = true;
				}
				
				if (!dump) continue;
				
				std::string single_transaction_str;
				_db_block_cache->Get(rocksdb::ReadOptions(), single_transaction_key, &single_transaction_str);
				_db_block_cache->Delete(rocksdb::WriteOptions(), single_transaction_key);
				_receipt_count.erase(single_transaction_key);
				transactions_str.push_back(std::move(single_transaction_str));
			}
		}
		
		//deserialization
		
		output.reserve(transactions_str.size());
		std::mutex insert_lock;
		auto_multi_thread::ParallelExecution(std::thread::hardware_concurrency(), [&insert_lock, &output](uint32_t index, std::string& trans_str){
			std::stringstream ss;
			ss << trans_str;
			auto trans = deserialize_wrap<boost::archive::binary_iarchive, transaction>(ss);
			{
				std::lock_guard guard(insert_lock);
				output.push_back(std::move(trans));
			}
		}, transactions_str.size(), transactions_str.data());
        _db_block_cache->FlushWAL(true); //reduce memory consumption
#else
		std::vector<std::string> all_transaction_key_data;
		for (auto& [key, _]: _map_block_cache)
		{
			all_transaction_key_data.push_back(key);
		}
		for (auto& single_transaction_key: all_transaction_key_data)
		{
			bool dump = false;
			auto iter_receipt = _receipt_count.find(single_transaction_key);
			if (iter_receipt == _receipt_count.end())
			{
				dump = true;
			}
			else
			{
				if (iter_receipt->second >= dump_threshold)	dump = true;
			}
			if (!dump) continue;
			
			auto iter_block_cache = _map_block_cache.find(single_transaction_key);
			output.push_back(iter_block_cache->second);
			_map_block_cache.erase(iter_block_cache);
			_receipt_count.erase(iter_receipt);
		}
#endif
		_block_cache_size = 0;
		return output;
	}
#pragma endregion
	
private:
	class verified_transaction_item : i_hashable, i_json_serialization
	{
	public:
		std::string hash_sha256;
		std::string signature;
		std::string receipt_hash;
		transaction_receipt receipt;
		
		void to_byte_buffer(byte_buffer& target) const override
		{
			target.add(hash_sha256);
			target.add(signature);
			target.add(receipt_hash);
			receipt.to_byte_buffer(target);
		}
		
		[[nodiscard]] i_json_serialization::json to_json() const override
		{
			i_json_serialization::json output;
			output["hash_sha256"] = hash_sha256;
			output["signature"] = signature;
			output["receipt_hash"] = receipt_hash;
			output["receipt"] = receipt.to_json();
			
			return output;
		}
		
		void from_json(const i_json_serialization::json& json_target) override
		{
			hash_sha256 = json_target["hash_sha256"];
			signature = json_target["signature"];
			receipt_hash = json_target["receipt_hash"];
			receipt.from_json(json_target["receipt"]);
		}
		
	private:
		friend class boost::serialization::access;
		template<class Archive>
		void serialize(Archive & ar, const unsigned int version)
		{
			ar & hash_sha256;
			ar & signature;
			ar & receipt_hash;
			ar & receipt;
		}
	};
	
#if USE_ROCKSDB
	rocksdb::DB* _db_verified_transactions;
	std::filesystem::path  _db_path_sub_verified_transactions;
	rocksdb::DB* _db_block_cache;
	std::filesystem::path  _db_path_sub_block_cache;
	std::mutex _db_block_cache_lock;
#else
	std::unordered_map<std::string, verified_transaction_item> _map_verified_transactions;
	std::unordered_map<std::string, transaction> _map_block_cache;
#endif

	size_t _block_cache_size;
	std::unordered_map<std::string, int> _receipt_count;
};

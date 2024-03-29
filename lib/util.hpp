#pragma once

#include <vector>
#include <random>
#include <map>

#define GENERATE_GET(Var_name, get_name)    \
auto& get_name()                            \
{return Var_name;}                          \
[[nodiscard]] const auto& get_name() const  \
{return Var_name;}

#define GENERATE_READ(Var_name, get_name)   \
[[nodiscard]] const auto& get_name() const  \
{return Var_name;}

namespace util
{
	template <typename T>
	std::vector<T> vector_concatenate(const std::vector<T>& a, const std::vector<T>& b)
	{
		std::vector<T> output;
		output.reserve( a.size() + b.size() );
		output.insert( output.end(), a.begin(), a.end() );
		output.insert( output.end(), b.begin(), b.end() );
		return std::move(output);
	}
	
	template <typename T>
	void vector_twin_shuffle(std::vector<T>& data0, std::vector<T>& data1)
	{
		assert(data0.size() == data1.size());
		auto n = data0.end() - data0.begin();
		for (size_t i=0; i<n; ++i)
		{
			static std::random_device dev;
			static std::mt19937 rng(dev());
			std::uniform_int_distribution<size_t> distribution(0,n-1);
			
			size_t random_number = distribution(rng);
			data0[i].swap(data0[random_number]);
			data1[i].swap(data1[random_number]);
		}
	}
	
	template <typename T>
	class bool_setter
	{
	public:
		explicit bool_setter(T& value) : _value(value)
		{
			_value = true;
		}
		
		~bool_setter() noexcept
		{
			_value = false;
		}
		
		bool_setter(const bool_setter&) = delete;
		bool_setter& operator=(const bool_setter&) = delete;
	
	private:
		T& _value;
	};
	
	std::string get_random_str(int length = 20)
	{
		std::string randomStr;
		static std::random_device randomDevice;
		std::default_random_engine randomEngine(randomDevice());
		std::uniform_int_distribution<int> distribution(0, 10+26+26-1);
		
		for (int i = 0; i < length; i++)
		{
			const int tempValue = distribution(randomEngine);
			if (tempValue < 10)
			{
				randomStr.push_back('0' + tempValue);
				continue;
			}
			else if(tempValue < 36)
			{
				randomStr.push_back('a' + tempValue-10);
				continue;
			}
			else if (tempValue < 62)
			{
				randomStr.push_back('A' + tempValue-36);
				continue;
			}
			else
			{
				throw std::logic_error("Impossible path");
			}
		}
		return randomStr;
	}
    
    template<typename K, typename V>
    bool cmp_map_value_ascending(std::pair<K, V> &a, std::pair<K, V> &b)
    {
        return a.second < b.second;
    }
    
    template<typename K, typename V>
    bool cmp_map_value_descending(std::pair<K, V> &a, std::pair<K, V> &b)
    {
        return a.second > b.second;
    }
    
    template<typename K, typename V>
    std::vector<std::pair<K, V>> sort_map_according_to_value(std::map<K, V>& target, bool descending = false)
    {
        std::vector<std::pair<K, V>> output;
        for (auto &it: target)
        {
            output.push_back(it);
        }
        if (descending)
        {
            std::sort(output.begin(), output.end(), cmp_map_value_descending<K, V>);
        }
        else
        {
            std::sort(output.begin(), output.end(), cmp_map_value_ascending<K, V>);
        }

        return output;
    };
    
    template<typename Iter, typename RandomGenerator>
    Iter select_randomly(Iter start, Iter end, RandomGenerator &g)
    {
        std::uniform_int_distribution<> dis(0, std::distance(start, end) - 1);
        std::advance(start, dis(g));
        return start;
    }
    
    template<typename Iter>
    Iter select_randomly(Iter start, Iter end)
    {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return select_randomly(start, end, gen);
    }

    std::vector<std::string> split(const std::string &s, char delimiter) {
        std::vector<std::string> tokens;
        std::string token;
        std::istringstream tokenStream(s);

        while (std::getline(tokenStream, token, delimiter)) {
            tokens.push_back(token);
        }

        return tokens;
    }
}
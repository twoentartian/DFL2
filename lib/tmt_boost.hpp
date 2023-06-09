#pragma once

#include <boost/thread.hpp>
#include <boost/asio.hpp>
#include <thread>

#include <functional>
#include <mutex>
#include <atomic>

class tmt_boost
{
public:
	/** Use:
	 *
     tmt::ParallelExecution([](uint32_t index, uint32_t thread_index, Data& data...)
     {
         function body
     }, Count, data pointer...);
     *
     tmt::ParallelExecution(uint32_t worker, [](uint32_t index, uint32_t thread_index, Data& data...)
     {
         function body
     }, Count, data pointer...);
     */
	template <class Function, typename Count, typename ...T>
	static void ParallelExecution(Function func, Count count, T* ...data)
	{
		uint32_t totalThread = std::thread::hardware_concurrency();
		ParallelExecution(totalThread, func, count, data...);
	}
	
	template <class Function, typename Count, typename ...T>
	static void ParallelExecution(uint32_t totalThread, Function func, Count count, T* ...data)
	{
        boost::asio::io_service ioService;
        boost::thread_group threadpool;
        
        boost::asio::io_service::work work(ioService);
        
        for (int i = 0; i < totalThread; ++i)
        {
            threadpool.create_thread([ObjectPtr = &ioService] { ObjectPtr->run(); });
        }
        
        for (int i = 0; i < count; ++i)
        {
            ioService.post(boost::bind(func, data[i]...));
        }
        ioService.stop();
        
        threadpool.join_all();
	}
	
	template <class Function, typename Count, typename ...T>
	static void ParallelExecution_StepIncremental(Function func, Count count, T* ...data)
	{
		uint32_t totalThread = std::thread::hardware_concurrency();
		ParallelExecution_StepIncremental(totalThread, func, count, data...);
	}
	
	template <class Function, typename Count, typename ...T>
	static void ParallelExecution_StepIncremental(uint32_t totalThread, Function func, Count count, T* ...data)
	{
        boost::asio::io_service ioService;
        boost::thread_group threadpool;
        
        boost::asio::io_service::work work(ioService);
        
        for (int i = 0; i < totalThread; ++i)
        {
            threadpool.create_thread([ObjectPtr = &ioService] { ObjectPtr->run(); });
        }
        
        for (int i = 0; i < count; ++i)
        {
            ioService.post(boost::bind(func, std::forward<T>(data[i])...));
        }
        ioService.stop();
        
        threadpool.join_all();
	}
	
private:

};
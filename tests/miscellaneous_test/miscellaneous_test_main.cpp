#include <csignal>

#include <thread>
#include <cmath>
#include <sstream>
#include <filesystem>

#if USE_BACKTRACE
#define BOOST_STACKTRACE_USE_BACKTRACE
#include <boost/stacktrace.hpp>
#endif

#include <time_util.hpp>
#include <byte_buffer.hpp>
#include <time_util.hpp>
#include <thread_pool.hpp>
#include <measure_time.hpp>
#include <ml_layer.hpp>
#include <configure_file.hpp>
#include <duplicate_checker.hpp>
#include <performance_profiler.hpp>

#define BOOST_TEST_MAIN

#include <boost/test/included/unit_test.hpp>

BOOST_AUTO_TEST_SUITE (miscellaneous_test)
	
	BOOST_AUTO_TEST_CASE (time_converter_test)
	{
		auto now = time_util::get_current_utc_time();
		auto now_str = time_util::time_to_text(now);
		std::cout << "now str: " << now_str << std::endl;
		auto now_copy = time_util::text_to_time(now_str);
		BOOST_CHECK(now == now_copy);
	}
	
	BOOST_AUTO_TEST_CASE (thread_pool_test)
	{
		measure_time measureTime;
		measureTime.start();
		thread_pool pool;
		pool.set_delete_callback([](int64_t thread_id){
			std::cout << "delete:" << thread_id << std::endl;
		});
		pool.insert_thread([](){
			std::this_thread::sleep_for(std::chrono::milliseconds(100));
		});
		pool.insert_thread([](){
			std::this_thread::sleep_for(std::chrono::milliseconds(200));
		});
		pool.wait_for_close();
		measureTime.stop();
		BOOST_CHECK( (measureTime.measure_ms() - 200) < 5);
		std::cout << "time:" << measureTime.measure_ms() << "ms" << std::endl;
	}
	
	BOOST_AUTO_TEST_CASE (configuration_vector)
	{
		configuration_file configuration;
		configuration_file::json data;
		data["1"] = configuration_file::json::array({0,1,2,3,4});
		std::string data_str = data.dump();
		
		configuration.LoadConfigurationData(data_str);
		auto raw_data = *configuration.get_vec<int>("1");
		for (int i = 0; i < 5; ++i)
		{
			BOOST_CHECK(raw_data[i] == i);
		}
	}
	
	BOOST_AUTO_TEST_CASE (duplicate_checker_test)
	{
		duplicate_checker<int> checker(1, 1);
		BOOST_CHECK(checker.find(1) == false);
		checker.add(1);
		BOOST_CHECK(checker.find(1) == true);
		std::this_thread::sleep_for(std::chrono::seconds(2));
		BOOST_CHECK(checker.find(1) == false);
	}

	BOOST_AUTO_TEST_CASE (performance_profiler)
	{
		profiler_auto p1("p1");
		std::this_thread::sleep_for(std::chrono::seconds(1));
	}

#include <clustering/hierarchical_clustering.hpp>
    BOOST_AUTO_TEST_CASE (hierarchical_clustering)
    {
        std::vector<clustering::data_point_2d<float>> points = {{0.0,0.1}, {0.0,0.2}, {10.0,0.1},{10.0,0.1}};
        clustering::hierarchical_clustering temp(false, 1, 0.5);
        auto history = temp.process(points, 2);
        std::cout << clustering::hierarchical_clustering::cluster_merge_summary::print_all_summary(history).str() << std::endl;
    }

#include <tmt.hpp>
    BOOST_AUTO_TEST_CASE (tmt_test)
    {
        const int DATA_SIZE = 1000;
        int* data = new int[DATA_SIZE];
        for (int i = 0; i < DATA_SIZE; ++i)
        {
            data[i] = i;
        }

        std::atomic_bool pass = true;
        for (int i = 0; i < 5; ++i)
        {
            tmt::ParallelExecution_StepIncremental([&pass](uint32_t index, uint32_t thread_index, int& data)
                                                                {
                                                                    if (index != data)
                                                                    {
                                                                        pass = false;
                                                                    }
                                                                }, DATA_SIZE, data);
        }
        BOOST_CHECK(pass);
    }

#if USE_BACKTRACE
    void signalHandler(int sig_num)
    {
        std::cerr << boost::stacktrace::stacktrace();
        exit(sig_num);
    }

    BOOST_AUTO_TEST_CASE (signal_SIGSEGV_test)
    {
        signal(SIGSEGV, signalHandler);
        signal(SIGABRT, signalHandler);

        int *foo = (int*)-1; // make a bad pointer
        printf("%d\n", *foo);

    }
#endif

#include "CNet/CNet.hpp"
    BOOST_AUTO_TEST_CASE (CNet)
    {
        CNetwork<> net(20000); //Create a network of max size 20000 nodes
        net.create_albert_barabasi(1000, 10, 21354647); //Fill this size with AB model
        std::cout << net.mean_degree() << std::endl; //Compute mean degree

    }

BOOST_AUTO_TEST_SUITE_END()
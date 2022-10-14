#define BOOST_TEST_MAIN
#include <boost/test/included/unit_test.hpp>

#include <nlohmann/json.hpp>
#include <configure_file.hpp>

BOOST_AUTO_TEST_SUITE (json_test)
	
BOOST_AUTO_TEST_CASE (json_test_basic)
{
	using json = nlohmann::json;
	json json1;
	json1["pi"] = 3.141;
	json1["happy"] = true;
	std::string s = json1.dump();
	BOOST_CHECK(s == "{\"happy\":true,\"pi\":3.141}");
}

BOOST_AUTO_TEST_CASE (json_test_configuration)
{
    configuration_file file;
    configuration_file::json json1;
    json1["pi"] = 3.141;
    json1["happy"] = true;
    json1["extra"] = 3;
    std::vector<uint8_t> vector;
    vector.push_back(1);vector.push_back(2);vector.push_back(3);
    json1["vector_test"] = vector;
    file.SetDefaultConfiguration(json1);
    auto ret = file.LoadConfiguration("config.json");
    BOOST_CHECK(ret == configuration_file::configuration_file_return_code::NoError);
    {
        auto target = file.get<std::vector<uint8_t>>("vector_test");
        BOOST_CHECK(target);
        BOOST_CHECK((*target)[0] == 1);
        BOOST_CHECK((*target)[1] == 2);
        BOOST_CHECK((*target)[2] == 3);
    }
    {
        auto target = file.get<double>("pi");
        BOOST_CHECK(*target);
        BOOST_CHECK(*target == 3.141);
    }
    {
        auto target = file.get<bool>("happy");
        BOOST_CHECK(*target);
        BOOST_CHECK(*target == true);
    }
    {
        auto target = file.get<int>("extra");
        BOOST_CHECK(*target);
        BOOST_CHECK(*target == 3);
    }
}


BOOST_AUTO_TEST_SUITE_END()



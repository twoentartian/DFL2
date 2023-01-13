#pragma once

namespace os_info
{
	enum class os_type
	{
		unknown,
		linux,
		apple,
		windows,
		android,
		freebsd,
		other,
	};
	
	inline os_type get_os_type()
	{
#if __linux__
		return os_type::linux;
#elif __FreeBSD__
		return os_type::freebsd;
#elif __ANDROID__
		return os_type::android;
#elif __APPLE__
		return os_type::apple;
#elif _WIN32
		return os_type::windows;
#else
        return os_type::other;
#endif
	}
}


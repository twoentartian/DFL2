#pragma once

namespace compile_time_content
{
	const char* lenet_solver_memory_name = "lenet_solver_memory.prototxt";
	const char* lenet_solver_memory_content =
#include "generated/lenet_solver_memory.prototxt"
;

	const char* lenet_train_memory_name = "lenet_train_memory.prototxt";
	const char* lenet_train_memory =
#include "generated/lenet_train_memory.prototxt"
;
}
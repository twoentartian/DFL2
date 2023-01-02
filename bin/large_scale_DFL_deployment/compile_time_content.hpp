#pragma once

namespace compile_time_content
{
	const char* lenet_solver_memory_name = "lenet_solver_memory.prototxt";
	const char* lenet_solver_memory_content =
#include "generated/lenet_solver_memory.prototxt.generated"
;

	const char* lenet_train_memory_name = "lenet_train_memory.prototxt";
	const char* lenet_train_memory_content =
#include "generated/lenet_train_memory.prototxt.generated"
;

    const char* run_py_name = "run.py";
	const char* run_py_content =
#include "generated/run.py.generated"
;

    const char *analyze_result_py_name = "analyze_result.py";
    const char *analyze_result_py_content =
#include "generated/analyze_result.py.generated"
;
}
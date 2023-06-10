valgrind --tool=dhat --trace-children=yes --fair-sched=yes  ./DFL_simulator_opti

valgrind --tool=memcheck --leak-check=full --show-leak-kinds=all --trace-children=yes --fair-sched=yes  --log-file=valgrind-200.txt  ./DFL_simulator_opti


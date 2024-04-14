# DFL

## What is DFL?

DFL is a federated machine learning framework which uses blokchain as a proof of contributions to ML models, rather than a distributed ledger to records the aggregated models from clients.

## Features

- No centralized node, no centralized ML models.
- High performance blockchain system. Nearly zero overhead.

## [DFL1](https://github.com/twoentartian/DFL)  vs DFL2

- DFL2 supports [hunter](https://hunter.readthedocs.io/en/latest/index.html) to automatically download and configure the dependencies, which means you should be able to compile DFL2 on more platforms.
- DFL2 integrates large-scale DFL network infrastructures.

## Dependency

Install this package from package manager (such as `apt, yum`).

- libunwind (https://github.com/libunwind/libunwind)

It is optional to install these following dependencies with your package manager. If these dependencies are not found, hunter will download and compile them.
Hunter will only download necessary dependencies such as OpenBLAS, gflags etc. LMDB, CUDA support are not included within Hunter.

We recommend installing the following dependency from source code for better performance.

- openblas (https://github.com/xianyi/OpenBLAS)


## Getting started

### For deployment

1. Install CMake and GCC with C++17 support.

2. Compile DFL executable(the source code is in [DFL.cpp](https://github.com/twoentartian/DFL2/blob/main/bin/DFL/DFL.cpp), you can find everything you need in CMake), which will start a node in the DFL network. There are several tools that we recommend to build, they are listed below:

    - [Keys generator](https://github.com/twoentartian/DFL2/blob/main/bin/tool/generate_node_address.cpp): to generate private keys and public keys. These keys will be used in the configuration file.

3. Compile your own "reputation algorithm", which will define the way of updating ML models and updating the other nodes' reputation. This implementation is critical for different dataset distribution, malicious ratio situations. We provide four sample "reputation algorithm" [here](https://github.com/twoentartian/DFL/tree/main/bin/reputation_sdk/sample).

4. Run DFL executable, it should provide a sample configuration file for you.

5. Modify the configuration file as you wish, for example, peers, node address, private key, public key, etc. Notice that the batch_size and test_batch_size must be identical to the Caffe solver's configuration. Here is an [explaination file](https://github.com/twoentartian/DFL/blob/main/readme/config.json.txt) for the configuration.

6. DFL receive ML dataset by network, there is an executable file called [data_injector](https://github.com/twoentartian/DFL2/blob/main/bin/data_injector/mnist_data_injector.cpp) for MNIST dataset, use it to inject dataset to DFL. Current version of data_injector only supports I.I.D. dataset injection.

7. DFL will train the model once it receives enough dataset for training, and send it as a transaction to other nodes. The node will generate a block when generating enough transactions and perform FedAvg when receiving enough models from other nodes.

### For simulation

1. Perform step 1 in deployment.

2. Compile DFL_Simulator_mt for multi-threading optimization. Or DFL_Simulator_opti for less memory consumption but without "reputation algorithm" support.

   Some other tools:

    - [Dirichlet_distribution_generator_for_Non_IID dataset](https://github.com/twoentartian/DFL2/blob/main/bin/tool/dirichlet_distribution_config_generator.cpp), used to generate Dirichlet distribution. You can execute without any arguments it to get its usage.

    - [large_scale_simulation_generator](https://github.com/twoentartian/DFL2/blob/main/bin/tool/large_scale_simulation_generator.cpp), it can automatically generate a configuration file for many many nodes (the configuration file is over 3000+ lines, so you'd better use this tool if you want to simulate for over 20 nodes).

3. Run the simulator, it should generate a sample configuration file and execute simulation immediately. You can use Ctrl+C to exit.

4. Modify the configuration file with this [explanation file](https://github.com/twoentartian/DFL2/blob/main/readme/simulator_config.json.txt).

5. The simulator will automatically crate an output folder, whose name is the current time, in the executable path. The configuration file and reputation dll will also be copied to the output folder for easily reproduce the output.


### Reputation algorithm SDK API:

Please refer to this [link](https://github.com/twoentartian/DFL/tree/main/bin/reputation_sdk/sample) for sample reputation algorithm. The SDK API is not written yet.


### For more details

https://dl.acm.org/doi/10.1145/3600225
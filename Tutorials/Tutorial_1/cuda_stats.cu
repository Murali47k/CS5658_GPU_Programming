#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

int getCoresPerSM(int major, int minor) {

    switch (major) {

        case 2: // Fermi
            return (minor == 1) ? 48 : 32;

        case 3: // Kepler
            return 192;

        case 5: // Maxwell
            return 128;

        case 6: // Pascal
            if (minor == 0)
                return 64;
            return 128;

        case 7: // Volta / Turing
            return 64;

        case 8: // Ampere / Ada
            if (minor == 0)
                return 64;
            if (minor == 6 || minor == 9)
                return 128;
            return 64;

        case 9: // Hopper / Blackwell
            return 128;

        default:
            return 128;
    }
}


// Prints one formatted information line
void printInfo(const std::string& label, const std::string& value) {

    std::cout << std::left << std::setw(35) << label<< ": "<< value << "\n";
}


int main() {

    int deviceCount = 0;

    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {

        std::cerr << "CUDA Error: "
                  << cudaGetErrorString(error)
                  << "\n";

        return 1;
    }


    std::cout << "\n";
    std::cout << "CUDA DEVICE REPORT\n";

    printInfo("CUDA Devices Found",std::to_string(deviceCount));

    // DEVICE INFORMATION

    for (int i = 0; i < deviceCount; ++i) {

        cudaDeviceProp prop;

        error = cudaGetDeviceProperties(&prop, i);

        if (error != cudaSuccess) {

            std::cerr << "\nError getting properties for device "
                      << i << ": "
                      << cudaGetErrorString(error)
                      << "\n";

            continue;
        }


        int coresPerSM =getCoresPerSM(prop.major, prop.minor);

        int totalCores =prop.multiProcessorCount * coresPerSM;

        // DEVICE HEADER

        std::cout << "\n\n";
        std::cout << "------------------------------------------------------------\n";
        std::cout << "                      DEVICE " << i << "\n";
        std::cout << "------------------------------------------------------------\n";

        // GPU OVERVIEW

        std::cout << "\n[ GPU OVERVIEW ]\n";
        std::cout << "------------------------------------------------------------\n";

        printInfo("GPU Name", prop.name);

        printInfo("Compute Capability",std::to_string(prop.major) + "." +std::to_string(prop.minor));

        printInfo("Global Memory (MB)",std::to_string(prop.totalGlobalMem / (1024 * 1024)) + " MB ");

        // COMPUTE RESOURCES

        std::cout << "\n[ COMPUTE RESOURCES ]\n";
        std::cout << "------------------------------------------------------------\n";

        printInfo("Streaming Multiprocessors",std::to_string(prop.multiProcessorCount));

        printInfo("CUDA Cores Per SM",std::to_string(coresPerSM));

        printInfo("Estimated Total CUDA Cores",std::to_string(totalCores));

        printInfo("Registers Per SM",std::to_string(prop.regsPerMultiprocessor));

        printInfo("Registers Per Block",std::to_string(prop.regsPerBlock));

        // THREAD CONFIGURATION

        std::cout << "\n[ THREAD CONFIGURATION ]\n";
        std::cout << "------------------------------------------------------------\n";

        printInfo("Warp Size",std::to_string(prop.warpSize) +" threads");

        printInfo("Max Threads Per Block",std::to_string(prop.maxThreadsPerBlock));

        printInfo("Max Threads Per SM",std::to_string(prop.maxThreadsPerMultiProcessor));

        printInfo(
            "Max Threads Dimension",
            std::to_string(prop.maxThreadsDim[0]) + " x " +
            std::to_string(prop.maxThreadsDim[1]) + " x " +
            std::to_string(prop.maxThreadsDim[2])
        );

        printInfo(
            "Max Grid Dimension",
            std::to_string(prop.maxGridSize[0]) + " x " +
            std::to_string(prop.maxGridSize[1]) + " x " +
            std::to_string(prop.maxGridSize[2])
        );

        // MEMORY

        std::cout << "\n[ MEMORY ]\n";
        std::cout << "------------------------------------------------------------\n";

        printInfo("Shared Memory Per Block",std::to_string(prop.sharedMemPerBlock / 1024) + " KB");

        printInfo("Shared Memory Per SM",std::to_string(prop.sharedMemPerMultiprocessor / 1024) + " KB");

        printInfo("Constant Memory",std::to_string(prop.totalConstMem / 1024) + " KB");

        printInfo("L2 Cache",std::to_string(prop.l2CacheSize / 1024) + " KB");

        printInfo("Memory Bus Width",std::to_string(prop.memoryBusWidth) +" bits");

        // CLOCK INFORMATION

        std::cout << "\n[ CLOCK INFORMATION ]\n";
        std::cout << "------------------------------------------------------------\n";

        printInfo("GPU Clock",std::to_string(prop.clockRate / 1000) + " MHz");

        printInfo("Memory Clock",std::to_string(prop.memoryClockRate / 1000) + " MHz");


        // HARDWARE FEATURES

        std::cout << "\n[ HARDWARE FEATURES ]\n";
        std::cout << "------------------------------------------------------------\n";

        printInfo("Concurrent Kernels",prop.concurrentKernels ? "Supported" : "Not Supported");

        printInfo("Unified Addressing",prop.unifiedAddressing ? "Supported" : "Not Supported");

        printInfo("ECC Support",prop.ECCEnabled ? "Enabled" : "Disabled");


    
    }


    return 0;
}
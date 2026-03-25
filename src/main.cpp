/*
 * main.cpp
 * ========
 * Entry point: parse CLI arguments, load the image, run the
 * requested filter implementation, measure time, save results.
 *
 * Example usage:
 *   ./image_filter --image photo.jpg --filter gaussian --kernel 7 --impl omp --threads 4
 */

#include "filters.h"
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cstdlib>
#include <stdexcept>

namespace fs = std::filesystem;

// ============================================================
// Print help text
// ============================================================
void print_help(const char* prog_name)
{
    std::cout << "\nImage Filter — Project 3\n"
              << "========================\n\n"
              << "Usage:\n"
              << "  " << prog_name << " [OPTIONS]\n\n"
              << "Required:\n"
              << "  --image   <path>                       Input image (PNG, JPG, ...)\n"
              << "  --filter  {box,gaussian,sharpen,sobel} Filter to apply\n"
              << "  --impl    {serial,omp,cuda}            Implementation to use\n\n"
              << "Optional:\n"
              << "  --kernel  <int>   Kernel size (odd: 3,7,11,...)  [default: 3]\n"
              << "  --threads <int>   OpenMP thread count             [default: 4]\n"
              << "  --block-x <int>   CUDA block width                [default: 16]\n"
              << "  --block-y <int>   CUDA block height               [default: 16]\n"
              << "  --repeats <int>   Timing repetitions              [default: 5]\n"
              << "  --output  <path>  Output directory                [default: results/]\n"
              << "  --csv     <path>  CSV log file                    [default: results.csv]\n"
              << "  --help            Show this message\n\n"
              << "Examples:\n"
              << "  ./image_filter --image img.jpg --filter gaussian --kernel 7 --impl serial\n"
              << "  ./image_filter --image img.jpg --filter sobel    --impl omp --threads 8\n"
              << "  ./image_filter --image img.jpg --filter box      --kernel 11 --impl cuda --block-x 32 --block-y 8\n\n";
}

// ============================================================
// All configuration in one struct
// ============================================================
struct Config {
    std::string image_path;
    std::string filter_name = "gaussian";
    std::string impl        = "serial";
    std::string output_dir  = "results";
    std::string csv_path    = "results.csv";

    int kernel_size = 3;
    int threads     = 4;
    int block_x     = 16;
    int block_y     = 16;
    int repeats     = 5;
};

// ============================================================
// parse_args — read the command-line into a Config struct
// ============================================================
Config parse_args(int argc, char* argv[])
{
    Config cfg;

    if (argc < 2) { print_help(argv[0]); exit(0); }

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") { print_help(argv[0]); exit(0); }

        // Macro to safely grab the next argument
        #define NEXT (i + 1 < argc ? argv[++i] : throw std::runtime_error("Missing value for " + arg))

        else if (arg == "--image")   cfg.image_path  = NEXT;
        else if (arg == "--filter")  cfg.filter_name = NEXT;
        else if (arg == "--impl")    cfg.impl        = NEXT;
        else if (arg == "--output")  cfg.output_dir  = NEXT;
        else if (arg == "--csv")     cfg.csv_path    = NEXT;
        else if (arg == "--kernel")  cfg.kernel_size = std::stoi(NEXT);
        else if (arg == "--threads") cfg.threads     = std::stoi(NEXT);
        else if (arg == "--block-x") cfg.block_x     = std::stoi(NEXT);
        else if (arg == "--block-y") cfg.block_y     = std::stoi(NEXT);
        else if (arg == "--repeats") cfg.repeats     = std::stoi(NEXT);
        else std::cerr << "Warning: unknown argument: " << arg << "\n";

        #undef NEXT
    }

    return cfg;
}

// ============================================================
// Convert filter name string → FilterType enum
// ============================================================
FilterType string_to_filter(const std::string& name)
{
    if (name == "box")      return FilterType::BOX;
    if (name == "gaussian") return FilterType::GAUSSIAN;
    if (name == "sharpen")  return FilterType::SHARPEN;
    if (name == "sobel")    return FilterType::SOBEL;
    throw std::runtime_error("Unknown filter: '" + name +
                             "'.  Options: box, gaussian, sharpen, sobel");
}

// ============================================================
// Log system info (CPU, cores, GPU, OS)
// ============================================================
void log_system_info()
{
    std::cout << "\n=== System Info ===\n";
    system("echo CPU: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d: -f2 | xargs)");
    system("echo Cores: $(nproc)");
    system("echo OS: $(uname -srm)");
    system("nvidia-smi --query-gpu=name,memory.total,compute_cap "
           "--format=csv,noheader 2>/dev/null || echo 'GPU: not available'");
    std::cout << "===================\n\n";
}

// ============================================================
// Append one result row to the CSV log
// ============================================================
void append_csv(const std::string& path,
                const std::string& filter, const std::string& impl,
                int rows, int cols, int kernel,
                int threads, int block_x, int block_y,
                double time_ms)
{
    bool is_new = !fs::exists(path);
    std::ofstream f(path, std::ios::app);

    if (is_new)
        f << "filter,impl,rows,cols,kernel,threads,block_x,block_y,time_ms,speedup\n";

    // Speedup is filled in by run_experiments.sh after all serial runs are done
    f << filter << "," << impl  << "," << rows  << "," << cols  << ","
      << kernel << "," << threads << "," << block_x << "," << block_y << ","
      << time_ms << ",1.0\n";
}

// ============================================================
// main
// ============================================================
int main(int argc, char* argv[])
{
    try {
        // --- Parse arguments ---
        Config cfg = parse_args(argc, argv);
        log_system_info();

        // --- Validate ---
        if (cfg.image_path.empty())
            throw std::runtime_error("--image <path> is required");
        if (cfg.kernel_size % 2 == 0)
            throw std::runtime_error("--kernel must be odd (3, 5, 7, 11, ...)");

        FilterType ft = string_to_filter(cfg.filter_name);

        // --- Load image ---
        int rows, cols;
        std::cout << "Loading image: " << cfg.image_path << "\n";
        auto image = load_image_gray(cfg.image_path, rows, cols);
        std::cout << "  Size: " << cols << " x " << rows << " pixels\n";

        // --- Create output directory ---
        fs::create_directories(cfg.output_dir);

        // --- Print run info ---
        std::cout << "\nFilter : " << cfg.filter_name
                  << "  |  Impl: " << cfg.impl
                  << "  |  Kernel: " << cfg.kernel_size << "x" << cfg.kernel_size;
        if (cfg.impl == "omp")  std::cout << "  |  Threads: " << cfg.threads;
        if (cfg.impl == "cuda") std::cout << "  |  Block: " << cfg.block_x << "x" << cfg.block_y;
        std::cout << "\n";

        // --- Run filter and measure time ---
        std::vector<float> output(rows * cols);
        double time_ms = 0.0;

        if (cfg.impl == "serial") {
            time_ms = measure_time_ms([&]() {
                apply_filter_serial(ft, image, output, rows, cols, cfg.kernel_size);
            }, cfg.repeats);

        } else if (cfg.impl == "omp") {
            time_ms = measure_time_ms([&]() {
                apply_filter_omp(ft, image, output, rows, cols, cfg.kernel_size, cfg.threads);
            }, cfg.repeats);

        } else if (cfg.impl == "cuda") {
            time_ms = measure_time_ms([&]() {
                apply_filter_cuda(ft, image, output, rows, cols,
                                  cfg.kernel_size, cfg.block_x, cfg.block_y);
            }, cfg.repeats);

        } else {
            throw std::runtime_error("Unknown --impl: '" + cfg.impl +
                                     "'.  Options: serial, omp, cuda");
        }

        std::cout << "Time: " << time_ms << " ms\n";

        // --- Save output image ---
        std::string out_img = cfg.output_dir + "/"
                            + cfg.filter_name + "_" + cfg.impl
                            + "_k" + std::to_string(cfg.kernel_size) + ".png";
        save_image(out_img, output, rows, cols);
        std::cout << "Saved: " << out_img << "\n";

        // --- Append to CSV ---
        append_csv(cfg.csv_path,
                   cfg.filter_name, cfg.impl, rows, cols,
                   cfg.kernel_size, cfg.threads, cfg.block_x, cfg.block_y,
                   time_ms);
        std::cout << "CSV:   " << cfg.csv_path << "\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}

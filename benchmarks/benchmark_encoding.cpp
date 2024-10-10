#include "benchmark_encoding.hpp"

using json = nlohmann::json;

/**
 * Main which calls multiple encoding tests.
 */
int main()
{
    try {
        sycl::queue q(sycl::gpu_selector_v);
        std::cout << "Running on "
                  << q.get_device().get_info<sycl::info::device::name>()
                  << std::endl;


        // Define the parameters for creating IdentityEncoding
        const json config{{EncodingParams::ENCODING, EncodingNames::GRID}, 
            {EncodingParams::N_LEVELS, 16}, 
            {EncodingParams::N_FEATURES_PER_LEVEL, 2}, 
            {EncodingParams::LOG2_HASHMAP_SIZE, 19}, 
            {EncodingParams::BASE_RESOLUTION, 16}, 
            {EncodingParams::PER_LEVEL_SCALE, 1.447269237440378}, 
            {EncodingParams::INTERPOLATION_METHOD, InterpolationType::Smoothstep}, 
            {EncodingParams::GRID_TYPE, GridType::Hash}, 
            {EncodingParams::N_DIMS_TO_ENCODE, 3}};
        benchmark_encoding<float, 64>(1 << 20, 3, 10, config, q);
        q.wait();

    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    } catch (...) {
        std::cout << "Caught some undefined exception." << std::endl;
        return 2;
    }


    return 0;
}
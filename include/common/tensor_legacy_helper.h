#include <torch/torch.h>

namespace xpu {
namespace dpcpp {

    torch::Tensor fromUSM(void *ptr, torch::Dtype dtype, std::vector<long> sizes, bool clone = true) {
        const torch::TensorOptions &options =
        torch::TensorOptions().dtype(dtype).device(torch::kXPU);
        if (clone) {
            return torch::from_blob(ptr, sizes, options).clone();
        }
        else {
            return torch::from_blob(ptr, sizes, options);
        }
    }

} // namespace dpcpp
} // namespace xpu

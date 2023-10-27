#include <iostream>
#include <vector>
#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>

int main() {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestONNXRuntime");
    Ort::SessionOptions session_options;
    Ort::Session session(env, "/model/dual_input_model.onnx", session_options);

    // Generate dummy input for two samples
    // std::vector<int32_t> seq_input_data = {
    //     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 3,  // Sample 1
    //     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 4   // Sample 2
    // };
    std::vector<int32_t> seq_input_data = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 4,  // Sample 1
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 3   // Sample 2
    };

    std::vector<int32_t> userinfo_data = {
        // 1, 1,  // Sample 1
        2, 2,  // Sample 1
        1, 1   // Sample 2
    };

    std::vector<int64_t> seq_input_shape = {2, 16};  // Batch size of 2, 16 features
    std::vector<int64_t> userinfo_shape = {2, 2};    // Batch size of 2, 2 features

    // Create input tensor objects
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    Ort::Value seq_input_tensor = Ort::Value::CreateTensor<int32_t>(memory_info, seq_input_data.data(), seq_input_data.size(), seq_input_shape.data(), 2);
    Ort::Value userinfo_tensor = Ort::Value::CreateTensor<int32_t>(memory_info, userinfo_data.data(), userinfo_data.size(), userinfo_shape.data(), 2);

    std::vector<const char*> input_names = {"seq_input", "userinfo"};
    // Create a vector to store inputs
    std::vector<Ort::Value> inputs;
    inputs.reserve(2);  // reserve space for two elements

    // Move Ort::Value objects into the vector
    inputs.push_back(std::move(seq_input_tensor));
    inputs.push_back(std::move(userinfo_tensor));

    // Run the model
    const char* output_names[] = {"output"};
    auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), inputs.data(), 2, output_names, 1);

    // Print out the result (assuming the output is a tensor)
    auto* floatarr = output_tensors[0].GetTensorMutableData<float>();
    for (int i = 0; i < 10; i++) {  // Assuming there are 5 classes for each of the 2 samples
        std::cout << "Sample " << (i/5) + 1 << ", Class " << i%5 << ": " << floatarr[i] << std::endl;
    }

    return 0;
}

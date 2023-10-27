#include <iostream>
#include <vector>
#include <cpu_provider_factory.h>
#include <onnxruntime_cxx_api.h>

int main() {
    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ModelTester");

    // Configure runtime options
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    // Register CPU provider
    Ort::Session session(env, "/model/simple_model.onnx", session_options);

    // Prepare model input
    float input[2] = {0.5, -0.5};  // Example input values
    std::vector<int64_t> input_shape = {1, 2};  // 1 is the batch size, 2 is the input dimension

    // Run model
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input, 2, input_shape.data(), 2);
    std::vector<const char*> input_names = {"input"};
    std::vector<const char*> output_names = {"output"};
    std::vector<Ort::Value> output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1, output_names.data(), 1);

    // Retrieve and print model output
    float* output = output_tensors[0].GetTensorMutableData<float>();
    std::cout << "Model output: " << *output << std::endl;

    return 0;
}
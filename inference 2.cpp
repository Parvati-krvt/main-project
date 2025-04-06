#include <iostream>
#include <vector>
#include <memory>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>

using namespace tflite;

std::unique_ptr<tflite::Interpreter> LoadModel(const std::string& model_path) {
    auto model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (!model) {
        std::cerr << "Failed to mmap model " << model_path << std::endl;
        exit(-1);
    }

    ops::builtin::BuiltinOpResolver resolver;
    std::unique_ptr<Interpreter> interpreter;
    InterpreterBuilder(*model, resolver)(&interpreter);
    if (!interpreter) {
        std::cerr << "Failed to construct interpreter for " << model_path << std::endl;
        exit(-1);
    }

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        std::cerr << "Failed to allocate tensors!" << std::endl;
        exit(-1);
    }

    return interpreter;
}

void FillInputTensor(tflite::Interpreter* interpreter, const std::vector<float>& data, int input_idx) {
    float* input = interpreter->typed_input_tensor<float>(input_idx);
    std::copy(data.begin(), data.end(), input);
}

std::vector<float> RunInference(tflite::Interpreter* interpreter) {
    if (interpreter->Invoke() != kTfLiteOk) {
        std::cerr << "Error during inference." << std::endl;
        exit(-1);
    }

    float* output = interpreter->typed_output_tensor<float>(0);
    int output_size = interpreter->tensor(interpreter->outputs()[0])->bytes / sizeof(float);
    return std::vector<float>(output, output + output_size);
}

int main() {
    // Load models
    auto rgb_interpreter = LoadModel("models/rgb_model.tflite");
    auto motion_interpreter = LoadModel("models/motion_model.tflite");
    auto lstm_interpreter = LoadModel("models/lstm_model.tflite");

    // Simulated input (replace with actual data from preprocessing)
    std::vector<float> rgb_input(30 * 7 * 7 * 1280, 0.5f);      // e.g., (30,7,7,1280) flattened
    std::vector<float> motion_input(30 * 7 * 7 * 64, 0.5f);     // e.g., (30,7,7,64) flattened

    // Step 1: Run RGB stream
    FillInputTensor(rgb_interpreter.get(), rgb_input, 0);
    auto rgb_output = RunInference(rgb_interpreter.get());  // e.g., shape (30, 128)

    // Step 2: Run Motion stream
    FillInputTensor(motion_interpreter.get(), motion_input, 0);
    auto motion_output = RunInference(motion_interpreter.get());  // e.g., shape (30, 64)

    // Step 3: Concatenate features for LSTM input
    std::vector<float> lstm_input;
    for (int i = 0; i < 30; ++i) {
        lstm_input.insert(lstm_input.end(), 
                          rgb_output.begin() + i * 128, 
                          rgb_output.begin() + (i + 1) * 128);
        lstm_input.insert(lstm_input.end(), 
                          motion_output.begin() + i * 64, 
                          motion_output.begin() + (i + 1) * 64);
    }

    // Step 4: Run LSTM classifier
    FillInputTensor(lstm_interpreter.get(), lstm_input, 0);
    auto final_output = RunInference(lstm_interpreter.get());

    std::cout << "Violence Probability: " << final_output[0] << std::endl;
    if (final_output[0] > 0.5) {
        std::cout << "Violence Detected!" << std::endl;
    } else {
        std::cout << "No Violence Detected." << std::endl;
    }

    return 0;
}

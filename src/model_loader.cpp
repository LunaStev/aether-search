#include <aethersearch/model_loader.h>
#include <tensorflow/c/c_api.h>
#include <tensorflow/c/tf_tensor.h>
#include <iostream>
#include <cstring>

#pragma comment(lib, "tenderflow")

void NoOpDeallocator(void* data, size_t length) {
    // pass
}

bool ModelLoader::LoadModel(const std::string& model_dir) {
    TF_Status* status = TF_NewStatus();

    TF_Graph* graph = TF_NewGraph();
    TF_SessionOptions* options = TF_NewSessionOptions();

    const char* tags[] = { "serve" };
    TF_Buffer* run_opts = nullptr;

    TF_Session* session = TF_LoadSessionFromSavedModel(
        options,
        run_opts,
        model_dir.c_str(),
        tags, 1,
        graph,
        nullptr,
        status
    );

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Failed to load model: " << TF_Message(status) << std::endl;
        return false;
    }

    this->graph = graph;
    this->session = session;

    std::cout << "Model loaded successfully from directory!" << std::endl;

    TF_DeleteSessionOptions(options);
    TF_DeleteStatus(status);
    return true;
}

static void write_little_endian(uint64_t value, void* target) {
    uint8_t* out = reinterpret_cast<uint8_t*>(target);
    for (int i = 0; i < 8; ++i) {
        out[i] = (value >> (i * 8)) & 0xFF;
    }
}

size_t EncodeVarint(uint64_t value, char* output) {
    size_t i = 0;
    while (value > 127) {
        output[i++] = static_cast<char>((value & 0x7F) | 0x80);
        value >>= 7;
    }
    output[i++] = static_cast<char>(value & 0x7F);
    return i;
}

std::vector<float> ModelLoader::Run(const std::string& input_text) {
    TF_Status* status = TF_NewStatus();

    char length_bytes[10];
    size_t length_size = EncodeVarint(input_text.size(), length_bytes);

    size_t str_tensor_size = sizeof(uint64_t) + length_size + input_text.size();

    int64_t dims[] = {1};
    TF_Tensor* input_tensor = TF_AllocateTensor(TF_STRING, dims, 1, str_tensor_size);
    void* tensor_data = TF_TensorData(input_tensor);

    memset(tensor_data, 0, sizeof(uint64_t));

    memcpy(static_cast<char*>(tensor_data) + sizeof(uint64_t), length_bytes, length_size);

    memcpy(static_cast<char*>(tensor_data) + sizeof(uint64_t) + length_size, input_text.data(), input_text.size());

    std::cout << "Input tensor ready" << std::endl;

    TF_Output input_op = { TF_GraphOperationByName((TF_Graph*)graph, "serving_default_inputs"), 0 };
    TF_Output output_op = { TF_GraphOperationByName((TF_Graph*)graph, "StatefulPartitionedCall_1"), 0 };

    if (!input_op.oper || !output_op.oper) {
        std::cerr << "Input/output node not found!" << std::endl;
        TF_DeleteTensor(input_tensor);
        TF_DeleteStatus(status);
        return {};
    }

    TF_Tensor* output_tensor = nullptr;

    TF_SessionRun(
        (TF_Session*)session,
        nullptr,
        &input_op, &input_tensor, 1,
        &output_op, &output_tensor, 1,
        nullptr, 0,
        nullptr,
        status
    );

    if (TF_GetCode(status) != TF_OK) {
        std::cerr << "Failed to execute session: " << TF_Message(status) << std::endl;
        TF_DeleteTensor(input_tensor);
        TF_DeleteStatus(status);
        return {};
    }

    std::cout << "세션 실행 완료" << std::endl;

    float* output_data = static_cast<float*>(TF_TensorData(output_tensor));
    std::vector<float> result(output_data, output_data + 512);

    TF_DeleteTensor(input_tensor);
    TF_DeleteTensor(output_tensor);
    TF_DeleteStatus(status);

    return result;
}
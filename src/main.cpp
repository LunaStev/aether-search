#include <iostream>
#include <aethersearch/model_loader.h>

#include <filesystem>

int main() {
    std::cout << "Hello, AetherSearch!" << std::endl;

    std::string base_path = std::filesystem::current_path().parent_path(); // 실행 시점 기준 경로
    std::string model_path = base_path + "/models/universal_encoder_v4";

    ModelLoader loader;
    if (!loader.LoadModel(model_path)) {
        std::cerr << "Failed to load model!" << std::endl;
        return 1;
    }

    std::cout << "Model load successful!" << std::endl;

    std::string input_text = "Hello, this is a test sentence.";
    std::vector<float> embedding = loader.Run(input_text);

    if (embedding.empty()) {
        std::cerr << "a failure of reasoning!" << std::endl;
        return 1;
    }

    std::cout << "Reasoning Success! Embedding Results:" << std::endl;

    for (size_t i = 0; i < embedding.size(); ++i) {
        std::cout << embedding[i] << ' ';
        if ((i + 1) % 8 == 0) std::cout << '\n';
    }

    return 0;
}

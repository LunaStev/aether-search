//
// Created by sobi1 on 2025-04-29.
//

#ifndef AETHERSEARCH_MODEL_LOADER_H
#define AETHERSEARCH_MODEL_LOADER_H

#include <string>
#include <vector>

class ModelLoader {
public:
    bool LoadModel(const std::string& model_path);
    std::vector<float> Run(const std::string& input_text);
private:
    void* session = nullptr;
    void* graph = nullptr;
};

#endif //AETHERSEARCH_MODEL_LOADER_H

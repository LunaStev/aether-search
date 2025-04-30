#include <cmath>
#include <iostream>
#include <aethersearch/model_loader.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <utility>

std::vector<std::pair<std::string, std::vector<float>>> LoadSentenceEmbeddings(
    const std::string& path, ModelLoader& loader
) {
    std::ifstream file(path);
    std::vector<std::pair<std::string, std::vector<float>>> results;

    if (!file.is_open()) {
        std::cerr << "Failed to open sentence file: " << path << std::endl;
        return results;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;

        std::vector<float> vec = loader.Run(line);
        if (vec.empty()) {
            std::cerr << "Conversion failed: " << line << std::endl;
            continue;
        }

        results.emplace_back(line, vec);
        std::cout << "Save complete: " << line << std::endl;
    }

    return results;
}

float CosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if (norm_a == 0.0f || norm_b == 0.0f) return 0.0f;
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
}

void Search(const std::vector<std::pair<std::string, std::vector<float>>>& database,
            const std::vector<float>& query_embedding) {
    float best_score = std::numeric_limits<float>::lowest();
    std::string best_sentence;

    for (const auto& pair : database) {
        const std::string& sentence = pair.first;
        const std::vector<float>& embedding = pair.second;

        float sim = CosineSimilarity(query_embedding, embedding);
        std::cout << "  Smilarity(" << sentence << "): " << sim << std::endl;
        if (sim > best_score) {
            best_score = sim;
            best_sentence = sentence;
        }
    }

    std::cout << "\nThe most similar sentence:\n";
    std::cout << "  \"" << best_sentence << "\" (Similarity: " << best_score << ")" << std::endl;
}

int main() {
    std::cout << "Hello, AetherSearch!" << std::endl;

    std::string base_path = std::filesystem::current_path().parent_path();
    std::string model_path = base_path + "/models/universal_encoder_v4";

    std::string sentences_file = (std::filesystem::current_path().parent_path() / "data/sentences.txt").string();

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

    auto database = LoadSentenceEmbeddings(sentences_file, loader);

    if (database.empty()) {
        std::cerr << "No database" << std::endl;
        return 1;
    }

    std::string user_query;
    std::cout << "\nType a sentence > ";
    std::getline(std::cin, user_query);

    std::vector<float> query_vec = loader.Run(user_query);
    if (query_vec.empty()) {
        std::cerr << "Failed Query embedding" << std::endl;
        return 1;
    }

    Search(database, query_vec);

    return 0;
}

#include "HNSW.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <sstream>

class VectorDataset {
public:
    std::vector<DataVector> set;
    
    void read_dataset(const std::string &filename) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            std::cerr << "ERROR: Cannot open file " << filename << std::endl;
            return;
        }
        
        std::string line;
        int lineNum = 0;
        bool headerSkipped = false;
        
        while (std::getline(file, line)) {
            lineNum++;
            
            if (!headerSkipped) {
                headerSkipped = true;
                continue;
            }
            
            if (line.empty()) continue;
            
            std::istringstream ss(line);
            DataVector dv;
            double val;
            std::string token;
            
            while (std::getline(ss, token, ',')) {
                try {
                    val = std::stod(token);
                    dv.push_back(val);
                } catch (const std::invalid_argument& e) {
                    // Skip non-numeric
                }
            }
            
            if (dv.size() > 0) {
                set.push_back(dv);
            }
        }
        
        file.close();
        std::cout << "Parsed " << set.size() << " vectors with dimension " << (set.size() > 0 ? set[0].size() : 0) << std::endl;
    }
    
    size_t size() { return set.size(); }
    DataVector& operator[](size_t idx) { return set[idx]; }
};

int main() {
    std::cout << "=== HNSW k-NN Search ===" << std::endl;
    
    std::string filename = "mnist-train.csv";
    std::ifstream testfile(filename);
    if (!testfile.good()) {
        std::cerr << "ERROR: File not found!" << std::endl;
        return 1;
    }
    testfile.close();
    
    std::cout << "Loading dataset..." << std::endl;
    VectorDataset trainData;
    trainData.read_dataset(filename);
    
    if(trainData.size() == 0) {
        std::cerr << "ERROR: Dataset empty!" << std::endl;
        return 1;
    }
    
    std::cout << "Dataset loaded: " << trainData.size() << " vectors, dimension " << trainData[0].size() << std::endl;
    
    HNSWGraph hnsw(16);  // M=16
    
    std::cout << "\nBuilding HNSW index..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    hnsw.buildIndex(trainData.set);
    
    auto buildEnd = std::chrono::high_resolution_clock::now();
    auto buildTime = std::chrono::duration_cast<std::chrono::milliseconds>(buildEnd - start);
    std::cout << "Index built in " << buildTime.count() << " ms" << std::endl;
    
    DataVector testQuery = trainData[100];
    std::cout << "\nSearching for 10 nearest neighbors..." << std::endl;
    
    auto searchStart = std::chrono::high_resolution_clock::now();
    auto results = hnsw.searchKNearest(testQuery, 10, 200);
    auto searchEnd = std::chrono::high_resolution_clock::now();
    
    auto searchTime = std::chrono::duration_cast<std::chrono::microseconds>(searchEnd - searchStart);
    
    std::cout << "\n=== Results ===" << std::endl;
    std::cout << "HNSW 10-NN Distances: ";
    for(double d : results) std::cout << d << " ";
    std::cout << "\n\nSearch time: " << searchTime.count() << " microseconds" << std::endl;
    std::cout << "Total time (build + search): " << (buildTime.count() + searchTime.count()/1000) << " ms" << std::endl;
    
    return 0;
}
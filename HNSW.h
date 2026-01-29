#ifndef HNSW_H
#define HNSW_H

#include <vector>
#include <unordered_set>
#include <set>
#include <queue>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>

class DataVector {
private:
    std::vector<double> v;
public:
    DataVector(size_t dimension = 0);
    ~DataVector();
    DataVector(const DataVector &other);
    DataVector &operator=(const DataVector &other);
    void setDimension(size_t dimension);
    DataVector operator+(const DataVector &other) const;
    DataVector operator-(const DataVector &other) const;
    double operator*(const DataVector &other) const;
    double norm() const;
    double dist(const DataVector &other) const;
    void push_back(double value);
    size_t size() const;
    double &operator[](int index);
    const double &operator[](int index) const;
};

class HNSWGraph {
private:
    struct Node {
        int id;
        std::vector<std::vector<int>> neighbors;  // neighbors[layer] = list of neighbor ids
        int maxLayer;
    };
    
    std::vector<Node> nodes;
    std::vector<DataVector> data;
    int M;                    // Max connections per node
    int maxM;                // Max for layer 0
    int maxM0;               // Max for layer > 0
    float ml;                // Normalization factor for layer assignment
    int maxLayer;            // Current max layer in graph
    int entryPoint;          // Entry point for search
    std::mt19937 rng;
    std::uniform_real_distribution<float> uniformDist;
    
    int getRandomLayer();
    std::vector<int> searchLayer(const DataVector &query, const std::vector<int> &entryPoints, int layer);
    std::vector<int> searchLayerGreedy(const DataVector &query, const std::vector<int> &entryPoints, int layer, int ef);
    void insertNode(const DataVector &data, int label, int layer);
    
public:
    HNSWGraph(int M = 16, float ml = 1.0 / log(2.0));
    ~HNSWGraph();
    
    void buildIndex(std::vector<DataVector> &dataset);
    std::vector<double> searchKNearest(const DataVector &query, int k, int ef = 200);
};

#endif
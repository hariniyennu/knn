#include "HNSW.h"
#include <fstream>
#include <sstream>

// --- DataVector Implementation ---
DataVector::DataVector(size_t dimension) { 
    v.resize(dimension, 0.0); 
}
DataVector::~DataVector() {}
DataVector::DataVector(const DataVector &other) : v(other.v) {}
DataVector& DataVector::operator=(const DataVector &other) { 
    if(this != &other) v = other.v; 
    return *this; 
}
void DataVector::setDimension(size_t dimension) { 
    v.assign(dimension, 0.0); 
}
void DataVector::push_back(double val) { 
    v.push_back(val); 
}
size_t DataVector::size() const { 
    return v.size(); 
}
double& DataVector::operator[](int i) { 
    return v[i]; 
}
const double& DataVector::operator[](int i) const { 
    return v[i]; 
}
double DataVector::norm() const { 
    double s = 0; 
    for(double x : v) s += x*x; 
    return sqrt(s); 
}
double DataVector::dist(const DataVector &other) const {
    double s = 0; 
    for(size_t i=0; i<v.size(); ++i) s += pow(v[i]-other.v[i], 2);
    return sqrt(s);
}
DataVector DataVector::operator-(const DataVector &other) const {
    DataVector res(v.size()); 
    for(size_t i=0; i<v.size(); ++i) res[i] = v[i]-other.v[i];
    return res;
}
double DataVector::operator*(const DataVector &other) const {
    double res = 0; 
    for(size_t i=0; i<v.size(); ++i) res += v[i]*other.v[i];
    return res;
}
DataVector DataVector::operator+(const DataVector &other) const {
    DataVector res(v.size());
    for(size_t i=0; i<v.size(); ++i) res[i] = v[i]+other.v[i];
    return res;
}

// --- HNSW Implementation ---
HNSWGraph::HNSWGraph(int M, float ml_val) 
    : M(M), maxM(M), maxM0(M*2), ml(ml_val), maxLayer(0), entryPoint(0), rng(42), uniformDist(0.0, 1.0) {
}

HNSWGraph::~HNSWGraph() {
    nodes.clear();
    data.clear();
}

int HNSWGraph::getRandomLayer() {
    return (int)(-log(uniformDist(rng)) * ml);
}

std::vector<int> HNSWGraph::searchLayer(const DataVector &query, const std::vector<int> &entryPoints, int layer) {
    std::vector<int> result;
    std::unordered_set<int> visited;
    std::priority_queue<std::pair<double, int>> candidates;  // max heap
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<std::pair<double, int>>> nearest;  // min heap
    
    double lowerBound = std::numeric_limits<double>::max();
    
    for (int ep : entryPoints) {
        double d = query.dist(data[ep]);
        lowerBound = std::min(lowerBound, d);
        candidates.push({-d, ep});
        nearest.push({d, ep});
        visited.insert(ep);
    }
    
    while (!candidates.empty()) {
        double currDist = -candidates.top().first;
        if (currDist > lowerBound) break;
        
        int curr = candidates.top().second;
        candidates.pop();
        
        for (int neighbor : nodes[curr].neighbors[layer]) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                double d = query.dist(data[neighbor]);
                
                if (d < lowerBound || nearest.size() < M) {
                    candidates.push({-d, neighbor});
                    nearest.push({d, neighbor});
                    lowerBound = std::min(lowerBound, d);
                }
            }
        }
    }
    
    while (!nearest.empty()) {
        result.push_back(nearest.top().second);
        nearest.pop();
    }
    
    return result;
}

std::vector<int> HNSWGraph::searchLayerGreedy(const DataVector &query, const std::vector<int> &entryPoints, int layer, int ef) {
    std::vector<int> result;
    std::unordered_set<int> visited;
    std::priority_queue<std::pair<double, int>> candidates;
    std::priority_queue<std::pair<double, int>, std::vector<std::pair<double, int>>, std::greater<std::pair<double, int>>> nearest;
    
    double lowerBound = std::numeric_limits<double>::max();
    
    for (int ep : entryPoints) {
        double d = query.dist(data[ep]);
        lowerBound = std::min(lowerBound, d);
        candidates.push({-d, ep});
        nearest.push({d, ep});
        visited.insert(ep);
    }
    
    while (!candidates.empty()) {
        double currDist = -candidates.top().first;
        if (currDist > lowerBound) break;
        
        int curr = candidates.top().second;
        candidates.pop();
        
        for (int neighbor : nodes[curr].neighbors[layer]) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                double d = query.dist(data[neighbor]);
                
                if (d < lowerBound || (int)nearest.size() < ef) {
                    candidates.push({-d, neighbor});
                    nearest.push({d, neighbor});
                    if ((int)nearest.size() > ef) {
                        lowerBound = nearest.top().first;
                        nearest.pop();
                    } else {
                        lowerBound = std::min(lowerBound, d);
                    }
                }
            }
        }
    }
    
    while (!nearest.empty()) {
        result.push_back(nearest.top().second);
        nearest.pop();
    }
    
    return result;
}

void HNSWGraph::buildIndex(std::vector<DataVector> &dataset) {
    std::cout << "Building HNSW index with " << dataset.size() << " points..." << std::endl;
    
    data = dataset;
    nodes.resize(dataset.size());
    
    for (size_t i = 0; i < dataset.size(); ++i) {
        nodes[i].id = i;
        int layer = getRandomLayer();
        nodes[i].maxLayer = layer;
        nodes[i].neighbors.resize(layer + 1);
        
        if (i == 0) {
            entryPoint = 0;
            maxLayer = layer;
        } else {
            std::vector<int> searchEps = {entryPoint};
            
            // Search and insert from top to target layer
            for (int lc = maxLayer; lc > layer; --lc) {
                auto nearest = searchLayer(dataset[i], searchEps, lc);
                if (!nearest.empty()) searchEps = {nearest[0]};
            }
            
            // Insert at all layers from layer to 0
            for (int lc = std::min(layer, maxLayer); lc >= 0; --lc) {
                auto candidates = searchLayerGreedy(dataset[i], searchEps, lc, 200);
                
                // Add bidirectional links
                int M = (lc == 0) ? maxM0 : maxM;
                
                for (int candidate : candidates) {
                    nodes[i].neighbors[lc].push_back(candidate);
                    nodes[candidate].neighbors[lc].push_back((int)i);
                    
                    // Prune if needed
                    if ((int)nodes[candidate].neighbors[lc].size() > M) {
                        nodes[candidate].neighbors[lc].pop_back();
                    }
                }
                
                searchEps = candidates;
            }
            
            // Update entry point if new max layer
            if (layer > maxLayer) {
                maxLayer = layer;
                entryPoint = i;
            }
        }
        
        if ((i + 1) % 5000 == 0) {
            std::cout << "Indexed " << (i + 1) << " points..." << std::endl;
        }
    }
    
    std::cout << "HNSW index built successfully!" << std::endl;
}

std::vector<double> HNSWGraph::searchKNearest(const DataVector &query, int k, int ef) {
    std::vector<int> searchEps = {entryPoint};
    
    // Search from top layer to layer 0
    for (int layer = maxLayer; layer > 0; --layer) {
        auto nearest = searchLayer(query, searchEps, layer);
        if (!nearest.empty()) searchEps = {nearest[0]};
    }
    
    // Final search at layer 0
    auto candidates = searchLayerGreedy(query, searchEps, 0, std::max(ef, k));
    
    std::vector<double> results;
    for (int i = 0; i < std::min(k, (int)candidates.size()); ++i) {
        results.push_back(query.dist(data[candidates[i]]));
    }
    
    return results;
}
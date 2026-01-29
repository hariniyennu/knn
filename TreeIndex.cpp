#include "TreeIndex.h"
#include <iostream>
#include <fstream>
#include <sstream>

// --- DataVector Implementation ---
DataVector::DataVector(size_t dimension) { 
    v.resize(dimension, 0.0); 
}
DataVector::~DataVector() {}
DataVector::DataVector(const DataVector &other) : v(other.v) {}
DataVector& DataVector::operator=(const DataVector &other) { 
    if(this != &other) v = other.v; return *this; 
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
    double res = 0; for(size_t i=0; i<v.size(); ++i) res += v[i]*other.v[i];
    return res;
}
DataVector DataVector::operator+(const DataVector &other) const {
    DataVector res(v.size());
    for(size_t i=0; i<v.size(); ++i) res[i] = v[i]+other.v[i];
    return res;
}

// --- VectorDataset Implementation ---
void VectorDataset::read_dataset(const std::string &filename) {
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
        
        // Skip header row (first line)
        if (!headerSkipped) {
            headerSkipped = true;
            continue;
        }
        
        // Skip empty lines
        if (line.empty()) continue;
        
        std::istringstream ss(line);
        DataVector dv;
        double val;
        
        // Parse comma-separated values
        std::string token;
        while (std::getline(ss, token, ',')) {
            try {
                val = std::stod(token);
                dv.push_back(val);
            } catch (const std::invalid_argument& e) {
                // Silently skip non-numeric values
            }
        }
        
        // Only add non-empty vectors
        if (dv.size() > 0) {
            set.push_back(dv);
        }
    }
    
    file.close();
    std::cout << "Parsed " << set.size() << " vectors with dimension " << (set.size() > 0 ? set[0].size() : 0) << std::endl;
}

// --- Tree Logic ---
void TreeIndex::clear(Node* node) {
    if (!node) return;
    clear(node->left); 
    clear(node->right);
    delete node;
}

// --- KDTreeIndex Implementation ---
void KDTreeIndex::Maketree(std::vector<DataVector> &dataset) {
    if (root) clear(root);  // Clear old tree if exists
    root = build(dataset.begin(), dataset.end());
}

TreeIndex::Node* KDTreeIndex::build(std::vector<DataVector>::iterator begin, std::vector<DataVector>::iterator end) {
    if (std::distance(begin, end) <= 100) {
        Node* n = new Node(); 
        n->isLeaf = true;
        n->points.assign(begin, end);
        return n;
    }
    
    // Find dimension with max spread
    int splitDim = 0;
    double maxSpread = -1;
    
    for(int i=0; i<begin->size(); ++i) {
        auto res = std::minmax_element(begin, end, [i](const DataVector& a, const DataVector& b) { 
            return a[i] < b[i]; 
        });

        auto min_it = res.first;
        auto max_it = res.second;

        double spread = (*max_it)[i] - (*min_it)[i];
        if(spread > maxSpread) { 
            maxSpread = spread; 
            splitDim = i; 
        }
    }
    
    // Sort and split
    std::sort(begin, end, [splitDim](const DataVector& a, const DataVector& b){ 
        return a[splitDim] < b[splitDim]; 
    });
    auto mid = begin + std::distance(begin, end)/2;
    
    Node* n = new Node();
    n->splitDim = splitDim;
    n->splitVal = (*mid)[splitDim];
    n->left = build(begin, mid);
    n->right = build(mid, end);
    return n;
}

std::vector<double> KDTreeIndex::searchKNearest(const DataVector &target, int k) {
    std::priority_queue<double> pq;
    searchRecursive(root, target, k, pq);
    std::vector<double> res;
    while(!pq.empty()) { 
        res.push_back(pq.top()); 
        pq.pop(); 
    }
    std::reverse(res.begin(), res.end());
    return res;
}

void KDTreeIndex::searchRecursive(Node* node, const DataVector &target, int k, std::priority_queue<double> &pq) {
    if (!node) return;
    if (node->isLeaf) {
        for(auto &p : node->points) {
            double d = target.dist(p);
            pq.push(d); 
            if(pq.size()>k) pq.pop();
        }
        return;
    }
    Node *nearer = (target[node->splitDim] <= node->splitVal) ? node->left : node->right;
    Node *farther = (target[node->splitDim] <= node->splitVal) ? node->right : node->left;

    searchRecursive(nearer, target, k, pq);
    if (pq.size() < k || std::abs(target[node->splitDim] - node->splitVal) < pq.top()) {
        searchRecursive(farther, target, k, pq);
    }
}

// --- RPTreeIndex Implementation ---
void RPTreeIndex::Maketree(std::vector<DataVector> &dataset) {
    if (root) clear(root);  // Clear old tree if exists
    root = build(dataset.begin(), dataset.end());
}

TreeIndex::Node* RPTreeIndex::build(std::vector<DataVector>::iterator begin, std::vector<DataVector>::iterator end) {
    if (std::distance(begin, end) <= 100) {
        Node* n = new Node();
        n->isLeaf = true;
        n->points.assign(begin, end);
        return n;
    }
    
    // Random Gaussian Direction
    static std::mt19937 gen(42);
    std::normal_distribution<double> dist(0, 1);
    DataVector dir(begin->size());
    for(size_t i=0; i<dir.size(); ++i) dir[i] = dist(gen);

    std::sort(begin, end, [&dir](const DataVector& a, const DataVector& b){ 
        return (a*dir) < (b*dir);
    });
    auto mid = begin + std::distance(begin, end)/2;

    Node* n = new Node();
    n->projDir = dir;
    n->splitVal = (*mid)*dir;
    n->left = build(begin, mid);
    n->right = build(mid, end);
    return n;
}

std::vector<double> RPTreeIndex::searchKNearest(const DataVector &target, int k) {
    std::priority_queue<double> pq;
    searchRecursive(root, target, k, pq);
    std::vector<double> res;
    while(!pq.empty()) { 
        res.push_back(pq.top()); 
        pq.pop(); 
    }
    std::reverse(res.begin(), res.end());
    return res;
}

void RPTreeIndex::searchRecursive(Node* node, const DataVector &target, int k, std::priority_queue<double> &pq) {
    if (!node) return;
    if (node->isLeaf) {
        for(auto &p : node->points) {
            double d = target.dist(p);
            pq.push(d);
            if(pq.size() > k) pq.pop();
        }
        return;
    }
    double proj = target * node->projDir;
    Node *nearer = (proj <= node->splitVal) ? node->left : node->right;
    Node *farther = (proj <= node->splitVal) ? node->right : node->left;

    searchRecursive(nearer, target, k, pq);
    if (pq.size() < k || std::abs(proj - node->splitVal) < pq.top()) {
        searchRecursive(farther, target, k, pq);
    }
}
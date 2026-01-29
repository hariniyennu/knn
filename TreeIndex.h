#ifndef TREEINDEX_H
#define TREEINDEX_H
#include <vector>
#include <queue>
#include <string>
#include <cmath>
#include <algorithm>
#include <random>

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

class VectorDataset {
public:
    std::vector<DataVector> set;
    void read_dataset(const std::string &filename);
    size_t size() { 
        return set.size(); 
    }
    DataVector& operator[](size_t idx) { 
        return set[idx];
    }
};

// Base Tree class
class TreeIndex {
public:
    struct Node {
        std::vector<DataVector> points; // For leaves
        int splitDim;                   // For KD-Tree
        double splitVal;                // Median or Delta
        DataVector projDir;             // For RP-Tree
        Node *left, *right;
        bool isLeaf;

        Node() : left(nullptr), right(nullptr), isLeaf(false), splitDim(-1), splitVal(0) {}
    };

    Node* root;
    TreeIndex() : root(nullptr) {}
    virtual ~TreeIndex() { 
        clear(root);
    }
    void clear(Node* node);
    virtual void Maketree(std::vector<DataVector> &dataset) = 0;
};

class KDTreeIndex : public TreeIndex {
    public:
        void Maketree(std::vector<DataVector> &dataset) override;
        std::vector<double> searchKNearest(const DataVector &target, int k);
    private:
        Node* build(std::vector<DataVector>::iterator begin, std::vector<DataVector>::iterator end);
        void searchRecursive(Node* node, const DataVector &target, int k, std::priority_queue<double> &pq);
};

class RPTreeIndex : public TreeIndex {
    public:
        void Maketree(std::vector<DataVector> &dataset) override;
        std::vector<double> searchKNearest(const DataVector &target, int k);
    private:
        Node* build(std::vector<DataVector>::iterator begin, std::vector<DataVector>::iterator end);
        void searchRecursive(Node* node, const DataVector &target, int k, std::priority_queue<double> &pq);
};

#endif
//
// Created by Liberty, Edo on 11/23/16.
//

#include "VectorUtils.h"
#include <algorithm>

using namespace std;

default_random_engine VectorUtils::generator;

std::vector<float> VectorUtils::randn(int d) {
    uniform_real_distribution<float> distribution(0.0, 5.0);

    std::vector<float> vector = std::vector<float>(d);
    for (int i = 0; i < d; i++) {
        vector[i] = distribution(generator);
    }
    return vector;
}

vector<vector<float>> VectorUtils::randm(int d, int n) {
    uniform_real_distribution<float> distribution(0.0, 5.0);

    vector<vector<float>> mat(n, vector<float>(d));

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++) {
            mat[i][j] = distribution(generator);
        }
    }
    return mat;
}

float VectorUtils::norm(vector<float> vector) {
    int d = vector.size();
    float sumOfSquare = 0.0;
    for (int i = 0; i < d; i++) {
        sumOfSquare += pow(vector[i], 2);
    }
    return sqrt(sumOfSquare);
}

bool VectorUtils::equal(std::vector<float> vector1, vector<float> vector2) {
    int d = vector1.size();
    if (vector2.size() != d)
        return false;
    float tollerance = 1E-5;
    double diffSquaredNorm = 0.0;
    double vector1SquaredNorm = 0.0;
    double vector2SquaredNorm = 0.0;
    for (int i = 0; i < d; i++) {
        vector1SquaredNorm += pow(vector1[i], 2);
        vector2SquaredNorm += pow(vector2[i], 2);
        diffSquaredNorm += pow(vector1[i] - vector2[i], 2);
    }
    return (sqrt(diffSquaredNorm) < tollerance * (sqrt(vector1SquaredNorm)) + sqrt(vector1SquaredNorm));
}

vector<int> VectorUtils::sample(int k, int d){
    if (d < k) return std::vector<int>();
    auto indicesSet = std::unordered_set<int>();

    uniform_int_distribution<int> uniformIntDistribution(0, d - 1); // guaranteed unbiased
    while (indicesSet.size() < k) {
        int index = uniformIntDistribution(generator);
        if (indicesSet.find(index) == indicesSet.end())
            indicesSet.insert(index);
    }
    auto indices = std::vector<int>(indicesSet.begin(), indicesSet.end());
    std::sort(indices.begin(), indices.end());
    return indices;
}


void VectorUtils::print(std::vector<int> indices) {
    int d = indices.size();
    std::cout << "[";
    if (d > 0)
        std::cout << indices[0];
    if (d > 1)
        for (int i = 1; i < d; i++) {
            std::cout << ", " << indices[i];
        }
    std::cout << "]\n";
}

void VectorUtils::print(std::vector<float> vector) {
    int d = vector.size();
    std::cout << "[";
    if (d > 0)
        std::cout << vector[0];
    if (d > 1)
        for (int i = 1; i < d; i++) {
            std::cout << ", " << vector[i];
        }
    std::cout << "]\n";
}

int VectorUtils::searchSortedIndices(int index, int *sortedIndices, int k) {
    if (k == 0) return 0;
    int left = 0;
    if (index <= sortedIndices[left]) return left;
    int right = k - 1;
    if (index > sortedIndices[right]) return right + 1;
    while (left + 1 < right) {
        int mid = (left + right) / 2;
        if (index <= sortedIndices[mid]) {
            right = mid;
            if (index > sortedIndices[right]) return right + 1;
        } else {
            left = mid;
            if (index <= sortedIndices[left]) return left;
        }
    }
    return left + 1;
}
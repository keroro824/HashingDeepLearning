//
// Created by Liberty, Edo on 11/23/16.
//

#ifndef CPP_VECTORUTILS_CPP_H
#define CPP_VECTORUTILS_CPP_H

#include <vector>
#include <iostream>
#include <random>
#include <unordered_set>
#include <math.h>

using namespace std;

class VectorUtils {
public:

    static default_random_engine generator;

    static vector<float> randn(int d);

    static vector<vector<float>> randm(int d, int n);

    static float norm(vector<float> vector);

    static bool equal(vector<float> vector1, vector<float> vector2);

    static vector<int> sample(int k, int d);

    static void print(vector<int> indices);

    static void print(vector<float> vector);

    static int searchSortedIndices(int index, int *sortedIndices, int k);

    //vector<int> sample2(int k, int d);
};


#endif //CPP_VECTORUTILS_CPP_H

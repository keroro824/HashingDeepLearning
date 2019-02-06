#include "Node.h"
#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include "Network.h"
#include <algorithm>
#include <map>
#include <climits>
#include <cstring>
#include <cfloat>

//#include"cnpy.h"
#include<complex>
#include<cstdlib>
#include<iostream>
#include<map>
#include<string>
#include "Config.h"
#define SVM


int Bucketsize = 100;
int *RangePow;
int *K;
int *L;
float *Sparsity;

int SMRangePow = 12;
int SMK = 1;
int SML = 100;
float SMSparsity  =  1;

int Batchsize = 1000;
int Rehash = 1000;
int InputDim = 784;
int totRecords = 60000;
int totRecordsTest = 10000;

int Adam = 0;
float Beta1 = 0.9;
float Beta2 = 0.999;
float Eps = 0.00000001;

float Lr = 0.01;
float LrFactor = 0;
int Epoch = 5;
int Stepsize = 20;

int *sizesOfLayers;
int numLayer = 3;
string trainData = "/Users/beidchen/Documents/work/SUBLIME/HashingDeepLearning/sgd/tf/MNIST_data/train-images-idx3-ubyte";
string testData = "/Users/beidchen/Documents/work/SUBLIME/HashingDeepLearning/sgd/tf/MNIST_data/t10k-images-idx3-ubyte";

string trainLabel = "/Users/beidchen/Documents/work/SUBLIME/HashingDeepLearning/sgd/tf/MNIST_data/train-labels-idx1-ubyte";
string testLabel = "/Users/beidchen/Documents/work/SUBLIME/HashingDeepLearning/sgd/tf/MNIST_data/t10k-labels-idx1-ubyte";
string Weights = "";
string savedWeights = "";
string logFile = "/search_labs/users/beidchen/workspace/SUBLIME/HashingDeepLearning/bgd_new/logs/log_wiki_acc";
using namespace std;
int globalTime = 0;

#define ALL(c) c.begin(), c.end()
#define FOR(i,c) for(typeof(c.begin())i=c.begin();i!=c.end();++i)
#define REP(i,n) for(int i=0;i<n;++i)
#define fst first
#define snd second
void endianSwap(unsigned int &x) {
    x = (x>>24)|((x<<8)&0x00FF0000)|((x>>8)&0x0000FF00)|(x<<24);
}
typedef vector<unsigned int> Image;



string trim(string& str)
{
    size_t first = str.find_first_not_of(' ');
    size_t last = str.find_last_not_of(' ');
    return str.substr(first, (last - first + 1));
}


void parseconfig(string filename)
{
    std::ifstream file(filename);
    if(!file)
    {
        cout<<"Error Config file not found: Given Filename "<< filename << endl;
    }
    std::string str;
    while (getline(file, str))
    {
        if (str == "")
            continue;

        std::size_t found = str.find("#");
        if (found != std::string::npos)
            continue;

        if (trim(str).length() < 3)
            continue;

        int index = str.find_first_of("=");
        string first = str.substr(0, index);
        string second = str.substr(index + 1, str.length());

        if (trim(first) == "Bucketsize")
        {
            Bucketsize = atoi(trim(second).c_str());
        }
        else if (trim(first) == "RangePow")
        {
            string str = trim(second).c_str();
            RangePow = new int[numLayer];
            char *mystring = &str[0];
            char *pch;
            pch = strtok(mystring, ",");
            int i=0;
            while (pch != NULL) {
                RangePow[i] = atoi(pch);
                pch = strtok(NULL, ",");
                i++;
            }
        }
        else if (trim(first) == "K")
        {
            string str = trim(second).c_str();
            K = new int[numLayer];
            char *mystring = &str[0];
            char *pch;
            pch = strtok(mystring, ",");
            int i=0;
            while (pch != NULL) {
                K[i] = atoi(pch);
                pch = strtok(NULL, ",");
                i++;
            }
        }
        else if (trim(first) == "L")
        {
            string str = trim(second).c_str();
            L = new int[numLayer];
            char *mystring = &str[0];
            char *pch;
            pch = strtok(mystring, ",");
            int i=0;
            while (pch != NULL) {
                L[i] = atoi(pch);
                pch = strtok(NULL, ",");
                i++;
            }
        }
        else if (trim(first) == "Sparsity")
        {
            string str = trim(second).c_str();
            Sparsity = new float[numLayer*2];
            char *mystring = &str[0];
            char *pch;
            pch = strtok(mystring, ",");
            int i=0;
            while (pch != NULL) {
                Sparsity[i] = atof(pch);
                pch = strtok(NULL, ",");
                i++;
            }
        }
        else if (trim(first) == "Batchsize")
        {
            Batchsize = atoi(trim(second).c_str());
        }
        else if (trim(first) == "Rehash")
        {
            Rehash = atoi(trim(second).c_str());
        }
        else if (trim(first) == "InputDim")
        {
            InputDim = atoi(trim(second).c_str());
        }
        else if (trim(first) == "totRecords")
        {
            totRecords = atoi(trim(second).c_str());
        }
        else if (trim(first) == "totRecordsTest")
        {
            totRecordsTest = atoi(trim(second).c_str());
        }
        else if (trim(first) == "Adam")
        {
            Adam = atoi(trim(second).c_str());
        }
        else if (trim(first) == "Beta1")
        {
            Beta1 = atof(trim(second).c_str());
        }
        else if (trim(first) == "Beta2")
        {
            Beta2 = atof(trim(second).c_str());
        }
        else if (trim(first) == "Eps")
        {
            Eps = atof(trim(second).c_str());
        }
        else if (trim(first) == "Lr")
        {
            Lr = atof(trim(second).c_str());
        }
        else if (trim(first) == "LrFactor")
        {
            LrFactor = atof(trim(second).c_str());
        }
        else if (trim(first) == "Epoch")
        {
            Epoch = atoi(trim(second).c_str());
        }
        else if (trim(first) == "Stepsize")
        {
            Stepsize = atoi(trim(second).c_str());
        }
        else if (trim(first) == "numLayer")
        {
            numLayer = atoi(trim(second).c_str());
        }
        else if (trim(first) == "logFile")
        {
            logFile = trim(second).c_str();
        }
        else if (trim(first) == "sizesOfLayers")
        {
            string str = trim(second).c_str();
            sizesOfLayers = new int[numLayer];
            char *mystring = &str[0];
            char *pch;
            pch = strtok(mystring, ",");
            int i=0;
            while (pch != NULL) {
                sizesOfLayers[i] = atoi(pch);
                pch = strtok(NULL, ",");
                i++;
            }
        }
        else if (trim(first) == "trainData")
        {
            trainData = trim(second).c_str();
        }
        else if (trim(first) == "testData")
        {
            testData = trim(second).c_str();
        }
        else if (trim(first) == "weight")
        {
            Weights = trim(second).c_str();
        }
        else if (trim(first) == "savedweight")
        {
            savedWeights = trim(second).c_str();
        }
        else if (trim(first) == "trainLabel")
        {
            trainLabel = trim(second).c_str();
        }
        else if (trim(first) == "testLabel")
        {
            testLabel = trim(second).c_str();
        }
        else
        {
            cout << "Error Parsing conf File at Line" << endl;
            cout << str << endl;
        }
    }
}

void EvalDataSVM(int numBatchesTest,  Network* _mynet, int iter){
    int totCorrect = 0;
    int debugnumber = 0;
    std::ifstream testfile(testData);
    string str;
//    std::getline( testfile, str );


    ofstream outputFile(logFile,  std::ios_base::app);
    for (size_t i = 0; i < numBatchesTest; i++) {
        int **records = new int *[Batchsize];
        float **values = new float *[Batchsize];
        int *sizes = new int[Batchsize];
        int **labels = new int *[Batchsize];
        int *labelsize = new int[Batchsize];
        int nonzeros = 0;
        int count = 0;
        vector<string> list;
        vector<string> value;
        vector<string> label;
        while (std::getline(testfile, str)) {

            char *mystring = &str[0];
            char *pch, *pchlabel;
            int track = 0;
            list.clear();
            value.clear();
            label.clear();
            pch = strtok(mystring, " ");
            pch = strtok(NULL, " :");
            while (pch != NULL) {
                if (track % 2 == 0)
                    list.push_back(pch);
                else if (track%2==1)
                    value.push_back(pch);
                track++;
                pch = strtok(NULL, " :");
            }

            pchlabel = strtok(mystring, ",");
            while (pchlabel != NULL) {
                label.push_back(pchlabel);
                pchlabel = strtok(NULL, ",");
            }

            nonzeros += list.size();
            records[count] = new int[list.size()];
            values[count] = new float[list.size()];
            labels[count] = new int[label.size()];
            sizes[count] = list.size();
            labelsize[count] = label.size();

            int currcount = 0;
            debugnumber++;
            vector<string>::iterator it;
            for (it = list.begin(); it < list.end(); it++) {
                records[count][currcount] = stoi(*it);
                currcount++;
            }
            currcount = 0;
            for (it = value.begin(); it < value.end(); it++) {
                values[count][currcount] = stof(*it);
                currcount++;
            }
            currcount = 0;
            for (it = label.begin(); it < label.end(); it++) {
                labels[count][currcount] = stoi(*it);
                currcount++;
            }

            count++;
            if (count >= Batchsize)
                break;
        }

        auto correctPredict = _mynet->predictClass(records, values, sizes, labels, labelsize);
        totCorrect += correctPredict;
        std::cout <<" iter "<< i<<" "<< totCorrect*1.0/(Batchsize*(i+1))<<std::endl;

        delete[] sizes;
        delete[] labels;
        for (size_t d = 0; d < Batchsize; d++) {
            delete[] records[d];
            delete[] values[d];
        }
        delete[] records;
        delete[] values;

    }
    testfile.close();

//    _mynet->saveWeights(savedWeights);
//
//    cnpy::npz_t arr;
//    if (LOADWEIGHT) {
//        arr = cnpy::npz_load(Weights);
//    }

    cout << "over all" << totCorrect * 1.0 / (numBatchesTest*Batchsize)<< endl;
    outputFile << iter<<" " <<globalTime/1000 << " " << totCorrect * 1.0 / (numBatchesTest*Batchsize) << endl;

}

void ReadDataSVM(int numBatches,  Network* _mynet, int epoch){
    std::ifstream file(trainData);
    float accumlogss = 0;
    std::string str;
//    std::getline( file, str );
    int totalTime = 0;
    for (size_t i = 0; i < numBatches; i++) {
        if((i+epoch*numBatches)%Stepsize==0) {
            EvalDataSVM(10, _mynet, epoch*numBatches+i);
        }
        int **records = new int *[Batchsize];
        float **values = new float *[Batchsize];
        int *sizes = new int[Batchsize];
        int **labels = new int *[Batchsize];
        int *labelsize = new int[Batchsize];
        int nonzeros = 0;
        int count = 0;
        vector<string> list;
        vector<string> value;
        vector<string> label;
        while (std::getline(file, str)) {
            char *mystring = &str[0];
            char *pch, *pchlabel;
            int track = 0;
            list.clear();
            value.clear();
            label.clear();
            pch = strtok(mystring, " ");
            pch = strtok(NULL, " :");
            while (pch != NULL) {
                if (track % 2 == 0)
                    list.push_back(pch);
                else if (track%2==1)
                    value.push_back(pch);
                track++;
                pch = strtok(NULL, " :");
            }

            pchlabel = strtok(mystring, ",");
            while (pchlabel != NULL) {
                label.push_back(pchlabel);
                pchlabel = strtok(NULL, ",");
            }

            nonzeros += list.size();
            records[count] = new int[list.size()];
            values[count] = new float[list.size()];
            labels[count] = new int[label.size()];
            sizes[count] = list.size();
            labelsize[count] = label.size();
            int currcount = 0;
            vector<string>::iterator it;
            for (it = list.begin(); it < list.end(); it++) {
                records[count][currcount] = stoi(*it);
                currcount++;
            }
            currcount = 0;
            for (it = value.begin(); it < value.end(); it++) {
                values[count][currcount] = stof(*it);
                currcount++;
            }

            currcount = 0;
            for (it = label.begin(); it < label.end(); it++) {
                labels[count][currcount] = stoi(*it);
                currcount++;
            }

            count++;
            if (count >= Batchsize)
                break;
        }

        bool rehash = false;
        if ((epoch*numBatches+i)%(Rehash/Batchsize) == (Rehash/Batchsize-1)){
            if(Mode==1 || Mode==4) {
                rehash = true;
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();


        auto logloss = _mynet->ProcessInput(records, values, sizes, labels, labelsize, epoch * numBatches + i,
                                            rehash);

        auto t2 = std::chrono::high_resolution_clock::now();

        int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
//        std::cout << " iter " << epoch * numBatches + i << "each point takes" << 1.0 * timeDiffInMiliseconds
//                  << std::endl;
        globalTime+= timeDiffInMiliseconds;

        delete[] sizes;

        for (size_t d = 0; d < Batchsize; d++) {
            delete[] records[d];
            delete[] values[d];
            delete[] labels[d];
        }
        delete[] records;
        delete[] values;
        delete[] labels;

    }
    file.close();

}



//void EvalDataq2bn(int numBatchesTest,  Network* _mynet){
//    int totCorrect = 0;
//    int debugnumber = 0;
//    std::ifstream testfile(testData);
//    string str;
//    for (size_t i = 0; i < numBatchesTest; i++) {
//        int **records = new int *[Batchsize];
//        float **values = new float *[Batchsize];
//        int *sizes = new int[Batchsize];
//        int *labels = new int[Batchsize];
//        int nonzeros = 0;
//        int count = 0;
//        vector<string> list;
//        vector<string> value;
//        while (std::getline(testfile, str)) {
//
//            char *mystring = &str[0];
//            char *pch;
//            pch = strtok(mystring, " ");
//            labels[count] = stoi(pch);
//            int track = 0;
//            list.clear();
//            value.clear();
//            while (pch != NULL) {
//                if (track!=0){
//                    list.push_back(pch);
//                }
//                pch = strtok(NULL, ", ");
//
//                track++;
//            }
//            nonzeros += list.size();
//            records[count] = new int[list.size()];
//            values[count] = new float[list.size()];
//            if (list.size()!=value.size()){
//
//            }
//
//            sizes[count] = list.size();
//            int currcount = 0;
//            debugnumber++;
//            vector<string>::iterator it;
//            for (it = list.begin(); it < list.end(); it++) {
//                records[count][currcount] = stoi(*it);
//                values[count][currcount] = 1;
//                currcount++;
//            }
//            count++;
//            if (count >= Batchsize)
//                break;
//        }
//        auto t1 = std::chrono::high_resolution_clock::now();
//        auto correctPredict = _mynet->predictClass(records, values, sizes, labels);
//        auto t2 = std::chrono::high_resolution_clock::now();
//        int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//        totCorrect += correctPredict;
//        std::cout <<" iter "<< i<<" "<< totCorrect*1.0/(Batchsize*(i+1))<<std::endl;
//        std::cout << " each point takes" << 1.0 * timeDiffInMiliseconds << std::endl;
//
//
//
//
//        delete[] sizes;
//        delete[] labels;
//        for (size_t d = 0; d < Batchsize; d++) {
//            delete[] records[d];
//            delete[] values[d];
//        }
//        delete[] records;
//        delete[] values;
//
//    }
//    testfile.close();
//    cout << "over all" << totCorrect * 1.0 / (numBatchesTest*Batchsize) << endl;
//
//}
//
//
//void ReadDataq2bn(int numBatches,  Network* _mynet, int e){
//    std::ifstream file(trainData);
//    float accumlogss = 0;
//    std::string str;
//    for (size_t i = 0; i < numBatches; i++) {
////        if(i%Stepsize==0) {
////            EvalDataq2bn(10, _mynet);
////        }
//        int **records = new int *[Batchsize];
//        float **values = new float *[Batchsize];
//        int *sizes = new int[Batchsize];
//        int *labels = new int[Batchsize];
//        int nonzeros = 0;
//        int count = 0;
//        vector<string> list;
//        vector<string> value;
//        while (std::getline(file, str)) {
//            char *mystring = &str[0];
//            char *pch;
//            pch = strtok(mystring, " ");
//            labels[count] = stoi(pch);
//            int track = 0;
//            list.clear();
//            value.clear();
//            while (pch != NULL) {
//                if (track!=0){
//                    list.push_back(pch);
//                }
//                pch = strtok(NULL, ", ");
//
//                track++;
//            }
//            nonzeros += list.size();
//            records[count] = new int[list.size()];
//            values[count] = new float[list.size()];
//            if (list.size()!=value.size()){
//
//            }
//
//            sizes[count] = list.size();
//            int currcount = 0;
//            vector<string>::iterator it;
//            for (it = list.begin(); it < list.end(); it++) {
//                records[count][currcount] = stoi(*it);
//                values[count][currcount] = 1;
//                currcount++;
//            }
//            count++;
//            if (count >= Batchsize)
//                break;
//        }
//        bool rehash = false;
//        if ((e*numBatches+i)%(Rehash/Batchsize) == (Rehash/Batchsize-1)){
//            rehash = true;
//        }
//
//        auto t1 = std::chrono::high_resolution_clock::now();
//        auto logloss = _mynet->ProcessInput(records, values, sizes, labels, e*numBatches + i, rehash);
//        auto t2 = std::chrono::high_resolution_clock::now();
//        int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
//        std::cout <<" iter "<< e*numBatches+ i<< " each point takes" << 1.0 * timeDiffInMiliseconds << std::endl;
//
//
//        delete[] sizes;
//        delete[] labels;
//        for (size_t d = 0; d < Batchsize; d++) {
//            delete[] records[d];
//            delete[] values[d];
//        }
//        delete[] records;
//        delete[] values;
//
//
//
//    }
//    file.close();
//}


int main(int argc, char* argv[])
{
    //***********************************
    // Parse Config File
    //***********************************
    parseconfig(argv[1]);


    //***********************************
    // Initialize Network
    //***********************************
    int numBatches = totRecords/Batchsize;
    int numBatchesTest = totRecordsTest/Batchsize;
    NodeType* layersTypes = new NodeType[numLayer];


    for (int i=0; i<numLayer-1; i++){
        layersTypes[i] = NodeType::ReLU;
    }
    layersTypes[numLayer-1] = NodeType::Softmax;

    cnpy::npz_t arr;
    if (LOADWEIGHT) {
        arr = cnpy::npz_load(Weights);
//        cnpy::NpyArray weightArr, biasArr;
//        float* weight, *bias;
//            weightArr = arr["weight_0"];
//            weight = weightArr.data<float>();
//            biasArr = arr["bias_0"];
//            bias = biasArr.data<float>();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    Network *_mynet = new Network(sizesOfLayers, layersTypes, numLayer, Batchsize, Lr, InputDim, K, L, RangePow, Sparsity, arr);
    auto t2 = std::chrono::high_resolution_clock::now();
    int timeDiffInMiliseconds = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "Network Initialization takes " << 1.0 * timeDiffInMiliseconds << std::endl;
//    _mynet->saveWeights(Weights);
//    EvalDataSVM(10, _mynet, 0);
//    exit(0);
    //***********************************
    // Start Training
    //***********************************

    for (int e=0; e< Epoch; e++) {
//#ifdef Q2BN
//        ReadDataq2bn(numBatches, _mynet, e);
//        EvalDataq2bn(numBatchesTest, _mynet);
//#endif
#ifdef SVM
        ofstream outputFile(logFile,  std::ios_base::app);
        outputFile<<"Epoch "<<e<<endl;
        ReadDataSVM(numBatches, _mynet, e);
        _mynet->saveWeights(savedWeights);
        if(e%5==4) {
            EvalDataSVM(numBatchesTest, _mynet, (e+1)*numBatches);
        }else{
            EvalDataSVM(100, _mynet, (e+1)*numBatches);
        }
#endif
//#ifdef EVAL
//        EvalDataq2bn(10, _mynet);
//#endif
    }

    delete [] RangePow;
    delete [] K;
    delete [] L;
    delete [] Sparsity;
    return 0;

}

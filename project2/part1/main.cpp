#include <iostream>
#include <fstream>
#include <string>
#include <climits>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <numeric>
#include <utility>

const int NUMLABELS = 10;
const int NUMBINS = 32;
const int RES = 8;
const double PSEUDOCOUNT = 0.0001;
const int DISCRETEMODE = 1;
const int CONTINUOUSMODE = 2;
const double PI = 3.1415926;

// read integer: big endian -> little endian
unsigned int readInt(std::ifstream& in) {
    char curr;
    unsigned int result = 0;
    for (int i = 0; i < sizeof(unsigned int) / sizeof(char); ++i) {
        in.read(&curr, 1);
        result <<= CHAR_BIT;
        result |= 0xFF & static_cast<unsigned int>(curr);
    }
    return result;
}

// read image as 1-d array
void readTrainingImage(std::ifstream& imagesFile, int label, int numRows, int numCols, std::vector<std::vector<std::vector<int>>>& record) {
    char currByte;
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            imagesFile.read(&currByte, 1);
            int value = 0xFF & static_cast<int>(currByte);
            record[label][i * numCols + j][value / RES]++;
        }
    }
}

// read test image and record value (bin) of each pixel
std::vector<int> readTestImage(std::ifstream& testFile, int numRows, int numCols, int mode) {
    std::vector<int> result(numRows * numCols, 0);
    char currByte;
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            testFile.read(&currByte, 1);
            int value = 0xFF & static_cast<int>(currByte);
            if (mode == DISCRETEMODE) {
                result[i * numCols + j] = value / RES;
            } else if (mode == CONTINUOUSMODE) {
                result[i * numCols + j] = value;
            }
        }
    }
    return result;
}

void countNumber(const std::vector<std::vector<std::vector<int>>>& record, std::vector<std::vector<int>>& count) {
    for (int i = 0; i < count.size(); ++i) {
        for (int j = 0; j < count[i].size(); ++j) {
            count[i][j] = 0;
            for (int k = 0; k < record[i][j].size(); ++k) {
                count[i][j] += record[i][j][k];
            }
        }
    }
}

// read image label
int readLabel(std::ifstream& labelsFile) {
    char input;    
    labelsFile.read(&input, 1);
    int target = 0xFF & static_cast<int>(input);
    return target;
}

// print images corresponding to 0 to 9 in Bayesian classifer
void printImagesDiscrete(const std::vector<std::vector<std::vector<int>>>& record, int numCols) {
    std::cout << std::endl;
    std::cout << "Imagination of numbers in Bayesian classifier: " << std::endl;
    for (int label = 0; label < NUMLABELS; ++label) {
        std::cout << std::endl;
        std::cout << label << ":";
        for (int elem = 0; elem < record[label].size(); ++elem) {
            auto mid = record[label][elem].begin() + NUMBINS / 2;
            int white = std::accumulate(record[label][elem].begin(), mid, 0);
            int black = std::accumulate(mid, record[label][elem].end(), 0);
            std::cout << (elem % numCols? ' ': '\n') << (black >= white); 
        }
        std::cout << std::endl;
    }
}

void printImagesContinuous(const std::vector<std::vector<double>>& mean, int numCols) {
    std::cout << std::endl;
    std::cout << "Imagination of numbers in Bayesian classifier: " << std::endl;
    for (int label = 0; label < NUMLABELS; ++label) {
        std::cout << std::endl;
        std::cout << label << ":";
        for (int elem = 0; elem < mean[label].size(); ++elem) {
            std::cout << (elem % numCols? ' ': '\n') << (mean[label][elem] >= 128); 
        }
        std::cout << std::endl;
    }
}

void normalization(std::vector<double>& data) {
    double total = std::accumulate(data.begin(), data.end(), 0.0);
    for (int i = 0; i < data.size(); ++i) {
        data[i] /= total;
    }
}

void printPosterior(const std::vector<double>& posterior) {
    std::cout << "Posterior (in log scale): " << std::endl;
    for (int i = 0; i < posterior.size(); ++i) {
        std::cout << i << ": " << std::fixed << std::setprecision(17) << posterior[i] << std::endl;
    }
}

void discreteMode(std::ifstream& trainingImagesFile, std::ifstream& trainingLabelsFile, std::ifstream& testImagesFile, std::ifstream& testLabelsFile) {
    unsigned int value;
    unsigned int numRows, numCols, numElems, numItems;
    /* read images in training set */
    value = readInt(trainingImagesFile);   // magic number
    value = readInt(trainingImagesFile);   // number of images
    numRows = readInt(trainingImagesFile); // number of rows
    numCols = readInt(trainingImagesFile); // number of columns
    numElems = numRows * numCols;

    /* read labels in training set */
    value = readInt(trainingLabelsFile);   // magic number
    numItems = readInt(trainingLabelsFile);   // number of items
    
    /* discrete: training */
    std::vector<int> prior(NUMLABELS, 0);
    std::vector<int> labels(numItems, 0);
    std::vector<std::vector<std::vector<int>>> record(NUMLABELS, std::vector<std::vector<int>>(numElems, std::vector<int>(NUMBINS, 0)));
    std::vector<std::vector<int>> count(NUMLABELS, std::vector<int>(numElems, 0));
    for (int i = 0; i < numItems; ++i) {
        labels[i] = readLabel(trainingLabelsFile);
        prior[labels[i]]++;
        readTrainingImage(trainingImagesFile, labels[i], numRows, numCols, record);
    }
    countNumber(record, count);
    std::vector<double> priorProb(NUMLABELS, 0.0);
    for (int i = 0; i < NUMLABELS; ++i) {
        priorProb[i] = static_cast<double>(prior[i]) / numItems;
    }    

    /* read images in test set */
    value = readInt(testImagesFile);   // magic number
    value = readInt(testImagesFile);   // number of images
    numRows = readInt(testImagesFile); // number of rows
    numCols = readInt(testImagesFile); // number of columns
    numElems = numRows * numCols;
    
    /* read labels in test set */
    value = readInt(testLabelsFile);   // magic number
    numItems = readInt(testLabelsFile);   // number of items
    
    /* discrete: classification */
    int error = 0;
    for (int i = 0; i < numItems; ++i) {
        if (i > 0) {
            std::cout << std::endl;
        }
        int answer = readLabel(testLabelsFile);
        std::vector<int> currImage = readTestImage(testImagesFile, numRows, numCols, DISCRETEMODE);
        std::vector<double> posterior(NUMLABELS, 0.0);
        for (int label = 0; label < NUMLABELS; ++label) {
            posterior[label] += log(priorProb[label]);
            for (int elem = 0; elem < numElems; ++elem) {
                int temp = record[label][elem][currImage[elem]];
                posterior[label] += log(std::max(PSEUDOCOUNT, static_cast<double>(temp)) / count[label][elem]);
            }
        }
        normalization(posterior);
        printPosterior(posterior);
        int decision = std::distance(posterior.cbegin(), std::min_element(posterior.cbegin(), posterior.cend()));
        std::cout << "Prediction: " << decision << ", Answer: " << answer << std::endl;
        if (decision != answer) {
            error++;
        }
    }

    /* print imagination of numbers in Baysian classifier */
    printImagesDiscrete(record, numCols);

    /* print error rate */
    std::cout << std::endl;
    std::cout << "Error rate: " << std::fixed << std::setprecision(4) << static_cast<double>(error) / numItems << std::endl;
}

void continuousMode(std::ifstream& trainingImagesFile, std::ifstream& trainingLabelsFile, std::ifstream& testImagesFile, std::ifstream& testLabelsFile) {
    unsigned int value;
    unsigned int numRows, numCols, numElems, numItems;
    /* read images in training set */
    value = readInt(trainingImagesFile);   // magic number
    value = readInt(trainingImagesFile);   // number of images
    numRows = readInt(trainingImagesFile); // number of rows
    numCols = readInt(trainingImagesFile); // number of columns
    numElems = numRows * numCols;

    /* read labels in training set */
    value = readInt(trainingLabelsFile);   // magic number
    numItems = readInt(trainingLabelsFile);   // number of items
    
    /* continuous: training */
    std::vector<int> prior(NUMLABELS, 0);
    std::vector<std::vector<double>> mean(NUMLABELS, std::vector<double>(numElems, 0.0));
    std::vector<std::vector<double>> variance(NUMLABELS, std::vector<double>(numElems, 0.0));
    for (int i = 0; i < numItems; ++i) {
        int currLabel = readLabel(trainingLabelsFile);
        prior[currLabel]++;
        for (int j = 0; j < numElems; ++j) {
            char currByte;
            trainingImagesFile.read(&currByte, 1);
            int value = 0xFF & static_cast<int>(currByte);
            mean[currLabel][j] += value;
            // variance[currLabel][j] += value * value;
        }
    }
    trainingImagesFile.clear();
    trainingLabelsFile.clear();
    trainingImagesFile.seekg(16, std::ios::beg);
    trainingLabelsFile.seekg(8, std::ios::beg);
    std::vector<double> priorProb(NUMLABELS, 0.0);
    for (int i = 0; i < NUMLABELS; ++i) {
        priorProb[i] = static_cast<double>(prior[i]) / numItems;
        for (int j = 0; j < numElems; ++j) {
            mean[i][j] /= prior[i];
            // variance[i][j] = std::max(PSEUDOCOUNT, (variance[i][j] / prior[i] - mean[i][j] * mean[i][j]));
        }
    }
    for (int i = 0; i < numItems; ++i) {
        int currLabel = readLabel(trainingLabelsFile);
        for (int j = 0; j < numElems; ++j) {
            char currByte;
            trainingImagesFile.read(&currByte, 1);
            int value = 0xFF & static_cast<int>(currByte);
            variance[currLabel][j] += pow(value - mean[currLabel][j], 2.0);
        }
    }
    for (int i = 0; i < NUMLABELS; ++i) {
        for (int j = 0; j < numElems; ++j) {
            variance[i][j] /= (prior[i] - 1);
            variance[i][j] = std::max(PSEUDOCOUNT, variance[i][j]);
        }
    }
    /* read images in test set */
    value = readInt(testImagesFile);   // magic number
    value = readInt(testImagesFile);   // number of images
    numRows = readInt(testImagesFile); // number of rows
    numCols = readInt(testImagesFile); // number of columns
    numElems = numRows * numCols;
    
    /* read labels in test set */
    value = readInt(testLabelsFile);   // magic number
    numItems = readInt(testLabelsFile);   // number of items
    
    /* continuous: classification */
    int error = 0;
    for (int i = 0; i < numItems; ++i) {
        if (i > 0) {
            std::cout << std::endl;
        }
        int answer = readLabel(testLabelsFile);
        std::vector<int> currImage = readTestImage(testImagesFile, numRows, numCols, CONTINUOUSMODE);
        std::vector<double> posterior(NUMLABELS, 0.0);
        for (int label = 0; label < NUMLABELS; ++label) {
            posterior[label] += log(priorProb[label]); 
           for (int elem = 0; elem < numElems; ++elem) {
                posterior[label] += log(1.0 / sqrt(2 * PI * variance[label][elem]));
                posterior[label] -= pow(currImage[elem] - mean[label][elem], 2.0) / (2.0 * variance[label][elem]);
            }
        }
        normalization(posterior);
        printPosterior(posterior);
        int decision = std::distance(posterior.cbegin(), std::min_element(posterior.cbegin(), posterior.cend()));
        std::cout << "Prediction: " << decision << ", Answer: " << answer << std::endl;
        if (decision != answer) {
            error++;
        }
    }

    /* print imagination of numbers in Baysian classifier */
    printImagesContinuous(mean, numCols);

    /* print error rate */
    std::cout << std::endl;
    std::cout << "Error rate: " << std::fixed << std::setprecision(4) << static_cast<double>(error) / numItems << std::endl;
}

int main() {
    /* get training set */
    std::string trainingImages("train-images.idx3-ubyte");
    std::string trainingLabels("train-labels.idx1-ubyte");
    std::ifstream trainingImagesFile(trainingImages.c_str(), std::ios::in | std::ios::binary);
    std::ifstream trainingLabelsFile(trainingLabels.c_str(), std::ios::in | std::ios::binary);
    if (!trainingImagesFile.is_open()) {
        std::cerr << "[Error] Cannot open file: " << trainingImages << std::endl;
        return 1;
    }
    if (!trainingLabelsFile.is_open()) {
        std::cerr << "[Error] Cannot open file: " << trainingImages << std::endl;
        return 2;
    }

    /* get test set */
    std::string testImages("t10k-images.idx3-ubyte");
    std::string testLabels("t10k-labels.idx1-ubyte");
    std::ifstream testImagesFile(testImages.c_str(), std::ios::in | std::ios::binary);
    std::ifstream testLabelsFile(testLabels.c_str(), std::ios::in | std::ios::binary);
    if (!testImagesFile.is_open()) {
        std::cerr << "[Error] Cannot open file: " << testImages << std::endl;
        return 3;
    }
    if (!testLabelsFile.is_open()) {
        std::cerr << "[Error] Cannot open file: " << testImages << std::endl;
        return 4;
    }

    int option = 0;
    while (option != 1 && option != 2) {
        std::cout << "Please enter mode of Baysian classifier (1: discrete; 2: continuous): ";
        std::cin >> option;
        if (option == 1) {
            discreteMode(trainingImagesFile, trainingLabelsFile, testImagesFile, testLabelsFile);
        } else if (option == 2) {
            continuousMode(trainingImagesFile, trainingLabelsFile, testImagesFile, testLabelsFile);
        } else {
            std::cerr << "[Error] Invalid number! Please try again!" << std::endl;
        }
    }
    
    /* close files */
    trainingImagesFile.close();
    trainingLabelsFile.close();
    testImagesFile.close();
    testLabelsFile.close();
    return 0;
}

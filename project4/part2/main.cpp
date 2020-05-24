#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <climits>
#include <algorithm>
#include <numeric>
#include <chrono>

const int NUMOFLABELS = 10;
const int THRESHOLD = 128;
const double TARGET = 20.0;

// read integer: big endian -> little endian
int readInt(std::ifstream& in) {
    char curr;
    int result = 0;
    for (int i = 0; i < sizeof(int) / sizeof(char); ++i) {
        in.read(&curr, 1);
        result <<= CHAR_BIT;
        result |= 0xFF & static_cast<int>(curr);
    }
    return result;
}

int readPixel(std::ifstream& in) {
    char curr;
    int result = 0;
    in.read(&curr, 1);
    result |= 0xFF & static_cast<int>(curr);
    return result;
}

double getDifference(const std::vector<std::vector<double>>& mu, const std::vector<std::vector<double>>& prevMu) {
    double diff = 0.0;
    for (int i = 0; i < mu.size(); ++i) {
        for (int j = 0; j < mu[i].size(); ++j) {
            diff += fabs(mu[i][j] - prevMu[i][j]);
        }
    }
    return diff;
}

void assignLabels(const std::vector<std::vector<int>>& data, const std::vector<int>& labels,
                  const std::vector<double>& pi, const std::vector<std::vector<double>>& mu, std::vector<int>& result) {
    std::vector<std::vector<int>> record(NUMOFLABELS, std::vector<int>(NUMOFLABELS, 0));
    for (int i = 0; i < data.size(); ++i) {
        std::vector<double> temp(pi);
        for (int j = 0; j < temp.size(); ++j) {
            for (int k = 0; k < data[i].size(); ++k) {
                temp[j] *= pow(mu[j][k], data[i][k]) * pow(1.0 - mu[j][k], 1.0 - data[i][k]);
            }
        }
        int largestProbIndex = std::distance(temp.cbegin(), std::max_element(temp.cbegin(), temp.cend()));
        record[labels[i]][largestProbIndex]++;
    }
    for (int i = 0; i < result.size(); ++i) {
        int maxV = 0;
        int maxLabel = 0;
        int maxClassIndex = 0;
        for (int m = 0; m < record.size(); ++m) {
            for (int n = 0; n < record[m].size(); ++n) {
                if (record[m][n] > maxV) {
                    maxV = record[m][n];
                    maxLabel = m;
                    maxClassIndex = n;
                }
            }
        }
        result[maxLabel] = maxClassIndex;
        for (int m = 0; m < record[maxLabel].size(); ++m) {
            record[maxLabel][m] = -1;
        }
        for (int m = 0; m < record.size(); ++m) {
            record[m][maxClassIndex] = -1;
        }
    }
}

void printImagination(const std::vector<std::vector<double>>& mu, const std::vector<int>& mapping, int numOfCols, bool isLabeled) {
    int numOfPixels = mu.front().size();
    for (int i = 0; i < NUMOFLABELS; ++i) {
        printf("\n%sclass %d: ", isLabeled ? "labeled " : "", i);
        for (int j = 0; j < numOfPixels; ++j) {
            if (j % numOfCols == 0) {
                printf("\n");
            }
            printf("%3d", mu[mapping[i]][j] >= 0.5 ? 1 : 0);
        }
        printf("\n");
    }
}

int getPrediction(const std::vector<int>& data, const std::vector<std::vector<double>>& mu, const std::vector<double>& pi) {
    std::vector<double> r(NUMOFLABELS, 0.0);
    for (int i = 0; i < NUMOFLABELS; ++i) {
        r[i] = pi[i];
        for (int j = 0; j < data.size(); ++j) {
            r[i] *= pow(mu[i][j], data[j]) * pow(1.0 - mu[i][j], 1.0 - data[j]);
        }
    }
    return std::distance(r.cbegin(), std::max_element(r.cbegin(), r.cend()));
}

void initParameters(std::vector<std::vector<double>>& mu, std::vector<double>& pi, std::vector<std::vector<double>>& r) {
    std::fill(pi.begin(), pi.end(), 1.0 / NUMOFLABELS);
    for (auto iter = r.begin(); iter != r.end(); ++iter) {
        std::fill(iter->begin(), iter->end(), 1.0 / NUMOFLABELS);
    }
    unsigned int seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < mu.size(); ++i) {
        for (int j = 0; j < mu[i].size(); ++j) {
            mu[i][j] = dist(gen);
        }
    }
}

bool checkCondition(const std::vector<double>& pi) {
    for (int i = 0; i < pi.size(); ++i) {
        if (pi[i] < 1e-3) {
            return false;
        }
    }
    return true;
}

int main() {
    /* get training set */
    std::string trainingImages("train-images.idx3-ubyte");
    std::string trainingLabels("train-labels.idx1-ubyte");
    std::ifstream trainingImageFile(trainingImages.c_str(), std::ios::in | std::ios::binary);
    std::ifstream trainingLabelFile(trainingLabels.c_str(), std::ios::in | std::ios::binary);
    if (!trainingImageFile.is_open()) {
        fprintf(stderr, "[Error] Cannot open file: %s\n", trainingImages.c_str());
        return 1;
    }
    if (!trainingLabelFile.is_open()) {
        fprintf(stderr, "[Error] Cannot open file: %s\n", trainingLabels.c_str());
        return 2;
    }
    
    /* read training data */
    int magicNum = 0, numOfImages = 0, numOfRows = 0, numOfCols = 0, numOfPixels = 0;
    magicNum = readInt(trainingLabelFile);
    numOfImages = readInt(trainingLabelFile);
    magicNum = readInt(trainingImageFile);
    numOfImages = readInt(trainingImageFile);
    numOfRows = readInt(trainingImageFile);
    numOfCols = readInt(trainingImageFile);
    numOfPixels = numOfRows * numOfCols;
 
    std::vector<std::vector<int>> data(numOfImages, std::vector<int>(numOfPixels, 0));
    std::vector<int> labels(numOfImages, 0);
    std::vector<std::vector<double>> mu(NUMOFLABELS, std::vector<double>(numOfPixels, 0.0));
    std::vector<std::vector<double>> prevMu(NUMOFLABELS, std::vector<double>(numOfPixels, 0.0));
    std::vector<double> pi(NUMOFLABELS, 0.0);
    std::vector<std::vector<double>> r(numOfImages, std::vector<double>(NUMOFLABELS, 0.0));
    std::vector<int> mapping(NUMOFLABELS, 0);

    for (int i = 0; i < mapping.size(); ++i) {
        mapping[i] = i;
    }
 
    for (int i = 0; i < numOfImages; ++i) {
        labels[i] = readPixel(trainingLabelFile);
        for (int j = 0; j < numOfPixels; ++j) {
            data[i][j] = static_cast<int>(readPixel(trainingImageFile) >= THRESHOLD);
        }
    }

    initParameters(mu, pi, r);

    int count = 0, condition = 0;
    double diff = 0.0;
    while (true) {
        // E step: evaluate reponsibilities
        for (int i = 0; i < numOfImages; ++i) {
            std::vector<double> record(pi);
            double currSum = 0.0;
            // printf("current evaluating image %d...\n", i);
            for (int j = 0; j < NUMOFLABELS; ++j) {
                for (int k = 0; k < numOfPixels; ++k) {
                    record[j] *= pow(mu[j][k], data[i][k]) * pow(1.0 - mu[j][k], 1.0 - data[i][k]);
                }
                currSum += record[j];
            }
            for (int j = 0; j < NUMOFLABELS; ++j) {
                if (currSum > 0) {
                    r[i][j] = record[j] / currSum;
                }
            }
        }

        prevMu = mu;

        // M step: maximize likelihood function
        std::vector<double> eNum(NUMOFLABELS, 0.0);
        for (int i = 0; i < numOfImages; ++i) {
            for (int j = 0; j < NUMOFLABELS; ++j) {
                eNum[j] += r[i][j];
            }
        }
        std::vector<std::vector<double>> temp(NUMOFLABELS, std::vector<double>(numOfPixels, 0.0));
        for (int i = 0; i < numOfImages; ++i) {
            for (int j = 0; j < NUMOFLABELS; ++j) {
                for (int k = 0; k < numOfPixels; ++k) {
                    temp[j][k] += r[i][j] * data[i][k];
                }
            }
        }
        for (int i = 0; i < NUMOFLABELS; ++i) {
            for (int j = 0; j < numOfPixels; ++j) {
                if (eNum[i] > 0.0) {
                    temp[i][j] /= eNum[i];
                }
            }
        }
        mu = temp;
        for (int i = 0; i < NUMOFLABELS; ++i) {
            pi[i] = eNum[i] / numOfImages;
        }

        // print the result of each iteration
        if (count > 0) {
            printf("\n-----------------------------------------------------------------------\n");
        }
        printImagination(mu, mapping, numOfCols, false);
        if (checkCondition(pi)) {
            ++condition;
        } else {
            condition = 0;
            initParameters(mu, pi, r);
        }
        diff = getDifference(mu, prevMu);
        printf("\nNo. of iteration(s): %d, Difference: %f\n", ++count, diff);
        if (diff < TARGET && condition >= 10 && std::accumulate(pi.cbegin(), pi.cend(), 0.0) > 0.95) {
            break;
        }
    }

    // print the labeled imagination
    printf("\n-----------------------------------------------------------------------\n");
    assignLabels(data, labels, pi, mu, mapping);
    printImagination(mu, mapping, numOfCols, true);

    std::vector<std::vector<std::vector<int>>> confusionMatrix(NUMOFLABELS, std::vector<std::vector<int>>(2, std::vector<int>(2, 0)));
    int numOfErrors = numOfImages;
    for (int i = 0; i < numOfImages; ++i) {
        int prediction = getPrediction(data[i], mu, pi);
        for (int j = 0; j < mapping.size(); ++j) {
            if (mapping[j] == prediction) {
                prediction = j;
                break;
            }
        }
        for (int j = 0; j < NUMOFLABELS; ++j) {
            if (labels[i] == j) {
                if (prediction == j) {
                    confusionMatrix[j][0][0]++;
                } else {
                    confusionMatrix[j][0][1]++;
                }
            } else {
                if (prediction == j) {
                    confusionMatrix[j][1][0]++;
                } else {
                    confusionMatrix[j][1][1]++;
                }
            }
        }
    }

    for (int i = 0; i < NUMOFLABELS; ++i) {
        printf("\n-----------------------------------------------------------------------\n");
        printf("\nConfusion matrix %d:\n", i);
        printf("                 Predict number %d   Predict not number %d\n", i, i);
        printf("Is number %d%22d%23d\n", i, confusionMatrix[i][0][0], confusionMatrix[i][0][1]);
        printf("Isn't number %d%19d%23d\n", i, confusionMatrix[i][1][0], confusionMatrix[i][1][1]);
        printf("\n");
        printf("Sensitivity (Successfully predict number %d): ", i);
        printf("%f\n", static_cast<double>(confusionMatrix[i][0][0]) / static_cast<double>(confusionMatrix[i][0][0] + confusionMatrix[i][0][1]));
        printf("Specificity (Successfully predict not number %d): ", i);
        printf("%f\n", static_cast<double>(confusionMatrix[i][1][1]) / static_cast<double>(confusionMatrix[i][1][0] + confusionMatrix[i][1][1]));
        numOfErrors -= confusionMatrix[i][0][0];
    }
    printf("\n");
    printf("Total iteration(s) to converge: %d\n", count);
    printf("Total error rate: %f\n", static_cast<double>(numOfErrors) / static_cast<double>(numOfImages));

    /* close files */
    trainingImageFile.close();
    trainingLabelFile.close();
    return 0;
}

#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <fstream>
#include <string>
#include <sstream>
#include <iterator>
#include <numeric>


using namespace std;

// Task 1: Implement and evaluate K-means and SOM


/**
 * Implement the kMeans and SOM algorithms and evaluate their performance
 * Testing of the algorithm functions is facilitated by having the same calling interface
 * For both, we use the Euclidean distance as the distance metric
*/


double euclidean_distance(double *a, double *b, int m_features){
    double dist = 0;
    for(int i=0; i<m_features; i++){
        dist += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return dist;
}


/**
 * @param data: input data set
 * @param n_samples: number of samples
 * @param m_features: number of features
 * @param K: number of clusters
 * @param max_iter: maximum number of iterations the K-means updates
 * @param assignment: output cluster assignment for each samplegit@github.com:R4phou/kth-il2233.git as a list of integer int[n_samples]
 * 
 * The idea consists to:
 * 1. Initialize the centroids (a centroid is a mean of a cluster)
 * 2. Update the assignment of each sample to the nearest centroid
 * 3. Update the centroids
 * 4. Repeat steps 2 and 3 until convergence or here max_iter
*/
void kmeans(int* assignment, int K, int max_iter, int n_samples, int m_features, double *data){
    // Initialize the centroids
    double centroids[K][m_features]; // Initialize an array of K centroids 
    for(int i=0; i<K; i++){
        for(int j=0; j<m_features; j++){
            // i*m_features + j is the index of the j-th feature of the i-th sample
            centroids[i][j] = data[i*m_features + j]; // Assign the first K samples as centroids
        }
    }

    // Initialize the assignment
    for(int i=0; i<n_samples; i++){
        assignment[i] = 0;
    }

    // Update the assignment
    for(int iter=0; iter<max_iter; iter++){
        for(int i=0; i<n_samples; i++){
            double min_dist = euclidean_distance(data + i*m_features, centroids[0], m_features);
            assignment[i] = 0;
            for(int j=1; j<K; j++){ // Find the nearest centroid and assign the sample to the corresponding cluster
                double dist = euclidean_distance(data + i*m_features, centroids[j], m_features);
                if(dist < min_dist){
                    min_dist = dist;
                    assignment[i] = j;
                }
            }
        }

        // Update the centroids

        // Count the number of samples in each cluster
        int count[K];
        for(int i=0; i<K; i++){
            count[i] = 0;
            for(int j=0; j<m_features; j++){
                centroids[i][j] = 0;
            }
        }

        // Sum the samples in each cluster
        for(int i=0; i<n_samples; i++){
            int cluster = assignment[i];
            for(int j=0; j<m_features; j++){
                centroids[cluster][j] += data[i*m_features + j];
            }
            count[cluster]++;
        }

        // Divide by the number of samples in each cluster
        for(int i=0; i<K; i++){
            for(int j=0; j<m_features; j++){
                centroids[i][j] /= count[i];
            }
        }
    }
}



struct t_pos{
    int x;
    int y;
};

/**
 * @param data: input data set
 * @param n_samples: number of samples
 * @param m_features: number of features
 * @param height: height of the SOM grid
 * @param width: width of the SOM grid
 * @param max_iter: number of iterations the SOM updates
 * @param assignment: output cluster assignment for each sample as a list of positions t_pos[n_samples]
 * @param lr: learning rate
 * @param sigma: rate that controls the weight update
 * 
 * There are 5 steps in the SOM algorithm:
 * 1. Initialize the weights with some small random numbers
 * 2. Competition: Each input will find its best matching unit using the Euclidean distance
 * 3. Cooperation: Update the weights of the best matching unit and its neighbors
 * 4. Adaptation: Decrease the learning rate and the neighborhood function
 * 5. Repeat steps 2-4 until convergence or max_iter
*/
void SOM(t_pos *assignment, double *data, int n_samples, int m_features, int height, int width, int max_iter, float lr, float sigma){
    double prev_weights[height][width][m_features];
    
    // Initialize the weights with some small random numbers
    double weights[height][width][m_features];
    for(int i=0; i<height; i++){
        for(int j=0; j<width; j++){
            for(int k=0; k<m_features; k++){
                weights[i][j][k] = (rand() % 100) / 100.0;
            }
        }
    }

    for(int iter=0; iter<max_iter; iter++){
        prev_weights[height][width][m_features] = weights[height][width][m_features];

        // Competition: Each input will find its best matching unit using the Euclidean distance
        for(int i=0; i<n_samples; i++){
            double min_dist = euclidean_distance(data + i*m_features, weights[0][0], m_features);
            assignment[i].x = 0;
            assignment[i].y = 0;
            for(int j=0; j<height; j++){
                for(int k=0; k<width; k++){
                    double dist = euclidean_distance(data + i*m_features, weights[j][k], m_features);
                    if(dist < min_dist){
                        min_dist = dist;
                        assignment[i].x = j;
                        assignment[i].y = k;
                    }
                }
            }
            // Assigment are the BMU (Best Matching Unit) coordinates

            // Cooperation: Update the weights of the best matching unit and its neighbors
            for(int j=0; j<height; j++){
                for(int k=0; k<width; k++){
                    double dist = (j - assignment[i].x) * (j - assignment[i].x) + (k - assignment[i].y) * (k - assignment[i].y);
                    double h = exp(-dist / (2 * sigma * sigma));
                    for(int l=0; l<m_features; l++){
                        weights[j][k][l] += lr * h * (data[i*m_features + l] - weights[j][k][l]);
                    }
                }
            }
        }
        
        if (prev_weights == weights){
            return;
        }
        // Adaptation: Decrease the learning rate and the neighborhood function
        lr *= exp(iter/(max_iter-1));
        sigma *= exp(iter/(max_iter-1));
    }
}


void print_to_file_kmeans(int* assignment, int n_samples, string filename){
    ofstream file;
    file.open(filename);
    for(int i=0; i<n_samples; i++){
        file << assignment[i] << endl;
    }
    file.close();
}


void print_to_file_som(t_pos* assignment, int n_samples, string filename){
    ofstream file;
    file.open(filename);
    for(int i=0; i<n_samples; i++){
        file << assignment[i].x << "," << assignment[i].y << endl;
    }
    file.close();
}

int main(){
    // Test the kmeans function
    int n_samples = 10;
    int m_features = 2;
    int K = 2;
    int max_iter = 100;
    double data[20] = {1, 1, 2, 2, 3, 3, 4, 4, 4, 4,  7, 7, 7, 7, 8, 8, 9, 9, 10, 10};
    int assignment[10];
    // The result should be [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]

    // Print the assignment for kmeans
    kmeans(assignment, K, max_iter, n_samples, m_features, data);
    print_to_file_kmeans(assignment, n_samples, "./output/kmeans_assignment.txt");

    // Test the SOM function
    int height = 2;
    int width = 2;
    max_iter = 100;
    float lr = 0.1;
    float sigma = 1.0;
    t_pos assignment2[10];
    // The result should be [(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]
    SOM(assignment2, data, n_samples, m_features, height, width, max_iter, lr, sigma);
    print_to_file_som(assignment2, n_samples, "./output/som_assignment.txt");

    return 0;
}
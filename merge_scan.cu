//#include "stdint.h"
//input/output array sizes
//inclusive sum
//changed

#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cub/cub.cuh>

#include <stdio.h> 

using namespace std;

#define NUM_STATES 4
#define NUM_CHARS  256
#define NUM_THREADS 128
#define NUM_LINES 2
#define INPUT_FILE "./input_file.txt"

typedef std::chrono::high_resolution_clock Clock;

__constant__ int     d_D[NUM_STATES * NUM_CHARS];
__constant__ uint8_t d_E[NUM_STATES * NUM_CHARS];


template <int states>
struct __align__(4) state_array{
    uint8_t v[states];

    __device__ state_array() {
        for(int i = 0; i < states; i++)
            v[i] = i;
    }

    __device__ void set_SA(int index, int x) {
	   v[index] = x;
    }

};

typedef state_array<NUM_STATES> SA;

__device__ void SA_copy(SA & a, SA &b) {
    for(int i = 0; i < NUM_STATES; i ++) 
        a.v[i] = b.v[i];
}

struct SA_op {
    __device__ SA operator()(SA &a, SA &b){
        SA c;
        for(int i = 0; i < NUM_STATES; i ++) 
            c.v[i] = b.v[a.v[i]];
        
        return c;
    }
};

 
__global__
void merge_scan (int num_chars, char* line, int* len_array, int array_len, int* output_array){


    typedef cub::BlockScan<SA, NUM_THREADS> BlockScan;
    typedef cub::BlockScan<int, NUM_THREADS> BlockScan2;

    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ typename BlockScan2::TempStorage temp_storage2;
    __shared__ SA prev_value;
    __shared__ int prev_sum;


    int block_num = blockIdx.x;
    int len = len_array[blockIdx.x];

    SA a = SA();
    SA_copy(prev_value , a);

    prev_sum = 0;


    for(int loop = threadIdx.x; loop < len; loop += NUM_THREADS) {
        if(loop < len) {

            if(loop % NUM_THREADS == 0) {
                SA_copy(a, prev_value);
            }

            else {   
                for(int i = 0; i < NUM_STATES; i++){
                    char c = line[loop + block_num * array_len];
                    int x = d_D[(int)(i* NUM_CHARS + c)];
                    a.set_SA(i, x);
                }
            }

            BlockScan(temp_storage).InclusiveScan(a, a, SA_op());
            __syncthreads();

            char c = line[loop + block_num * array_len];
            int state = a.v[0];
            int start = (int) d_E[(int) (NUM_CHARS * state + c)];
            int end;
            BlockScan2(temp_storage2).InclusiveSum(start, end);
            if(start == 1) 
                output_array[end + block_num * array_len - 1 + prev_sum] = loop;

            if((loop + 1) % NUM_THREADS == 0) {
                SA_copy(prev_value , a);
                prev_sum = end;
            }   
        }
        __syncthreads();


    }

}

__global__
void clear_array (int* input_array, int len) {

    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < len) {
        input_array[idx] = 0;
    }

}


//CPU functions

int     D[NUM_STATES][NUM_CHARS];
uint8_t E[NUM_STATES][NUM_CHARS];

void add_transition (int state, uint8_t input, int next_state) 
{
    D[state][input] = next_state;
}

void add_default_transition(int state, int next_state) 
{
    for (int i = 0; i < NUM_CHARS; i++) 
        D[state][i] = next_state;
}

void add_emission(int state, uint8_t input, uint8_t value) 
{
    E[state][input] = value;
}

void add_default_emission(int state, uint8_t value) 
{
    for (int i = 0; i < NUM_CHARS; i++) 
        E[state][i] = value;
}

void Dtable_generate() 
{
    for (int i = 0; i < NUM_STATES; i++) 
        add_default_transition(i ,i);
    
    add_default_transition(2 , 1);
    add_default_transition(3 , 0);

    add_transition(0, '[', 1);
    add_transition(1, '\\', 2);
    add_transition(1, ']', 0);
    add_transition(0, '\\', 3);
}

void Etable_generate() 
{
    for(int i = 0; i < NUM_STATES; i++) 
        add_default_emission(i, 0);
    
    add_emission(0, ',', 1);
}

int max_length()
{
    std::ifstream is(INPUT_FILE);   // open file
    string line;
    int length = 0; 

    while (getline(is, line)){
        if(length < line.length())
            length = line.length();
    }
    is.close();
    
    return length; 
}


int main() {

    Dtable_generate();
    Etable_generate();
    const int array_len = max_length(); 

    cudaMemcpyToSymbol(d_D, D, NUM_STATES * NUM_CHARS * sizeof(int));
    cudaMemcpyToSymbol(d_E, E, NUM_STATES * NUM_CHARS * sizeof(uint8_t));

    int* h_output_array = new int[NUM_LINES * array_len];

    //Memory allocation for kernel functions
    
    int* d_output_array;
    cudaMalloc((int**)&d_output_array, array_len * sizeof(int) * NUM_LINES);

    char* d_line;
    cudaMalloc((char**) &d_line, array_len * sizeof(char) * NUM_LINES);

    int* d_len_array;
    cudaMalloc((char**) &d_len_array, NUM_LINES * sizeof(int));


    std::ifstream is(INPUT_FILE);

    string line;
    char* input_strings = new char[NUM_LINES * array_len];

    int len_array[NUM_LINES];
    //int offset_array[NUM_LINES];
    int count = 0;

    //start timer
    auto t1 = Clock::now();
    while (getline(is, line)) 
    { 

        for(int i = 0; i < array_len; i++) {
            if(i < line.length())
                input_strings[count * array_len + i] = line[i];
            else
                input_strings[count * array_len + i] = 0;
        }

        len_array[count] = line.length();
        count++;

        if(count == NUM_LINES){
          
            cudaMemcpy(d_line, input_strings, array_len * sizeof(char) * NUM_LINES, cudaMemcpyHostToDevice);     
            cudaMemcpy(d_len_array, len_array, NUM_LINES * sizeof(int), cudaMemcpyHostToDevice);     


            dim3 dimGrid(NUM_LINES,1,1);
            dim3 dimBlock(NUM_THREADS,1,1);
            merge_scan<<<dimGrid, dimBlock>>>(1, d_line, d_len_array, array_len, d_output_array);
           
            cudaMemcpy(h_output_array, d_output_array, array_len  * sizeof(int) * NUM_LINES, cudaMemcpyDeviceToHost);
            
            for(int j = 0; j < NUM_LINES; j++) {
                for(int i = 0; i < array_len; i++) {
                    cout << h_output_array[i + j * array_len] << " ";

                }
                cout << endl;
                
            }
            clear_array<<<dimGrid, dimBlock>>>(d_output_array, array_len * NUM_LINES);
            count = 0;
        }
    }

    //if the total number of lines is not a multiple of NUM_LINES
    if(count != 0) {

        cudaMemcpy(d_line, input_strings, array_len * sizeof(char) * NUM_LINES, cudaMemcpyHostToDevice);     
        cudaMemcpy(d_len_array, len_array, NUM_LINES * sizeof(int), cudaMemcpyHostToDevice);     
        cudaDeviceSynchronize();

        dim3 dimGrid(count,1,1);
        dim3 dimBlock(NUM_THREADS,1,1);
        merge_scan<<<dimGrid, dimBlock>>>(1, d_line, d_len_array, array_len, d_output_array);
        cudaDeviceSynchronize();
        cudaMemcpy(h_output_array, d_output_array, array_len  * sizeof(int) * NUM_LINES, cudaMemcpyDeviceToHost);
        
        for(int j = 0; j < NUM_LINES; j++) {
                for(int i = 0; i < array_len; i++) {
                    cout << h_output_array[i + j * array_len] << " ";

                }
                cout << endl;   
        }
    }

    //end timer
    is.close();

    auto t2 = Clock::now();
    cout << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " microseconds" << endl;
    

    cudaFree(d_output_array);
    cudaFree(d_line);

    free(h_output_array);
    free(input_strings);


    return 0;
}


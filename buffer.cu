#include <cstdint>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <cub/cub.cuh>
#include <stdio.h> 

using namespace std;

#define NUM_STATES 3
#define NUM_CHARS  256
#define NUM_THREADS 612
#define NUM_LINES 30

#define BUFFER_SIZE = 2500
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
  //  typedef cub::BlockScan<int, NUM_THREADS> BlockScan2;

    __shared__ typename BlockScan::TempStorage temp_storage;
   // __shared__ typename BlockScan2::TempStorage temp_storage2;

    int block_num = blockIdx.x;

    int len = len_array[blockIdx.x];

    for(int loop = threadIdx.x; loop < len; loop += NUM_THREADS) {
        if(loop < len) {

            SA a = SA();
        	for(int i = 0; i < NUM_STATES; i++){
                char c = line[loop + block_num * array_len];
                int x = d_D[(int)(i* NUM_CHARS + c)];
        	    a.set_SA(i, x);
        	}

            BlockScan(temp_storage).InclusiveScan(a, a, SA_op());

            char c = line[loop + block_num * array_len];
            int state = a.v[0];
            output_array[loop + block_num * array_len ] = (int) d_E[(int) (NUM_CHARS * state + c)];
            /*
            int start = (int) d_E[(int) (NUM_CHARS * state + c)];
            int end;
            BlockScan2(temp_storage2).InclusiveSum(start, end);
            output_array[idx - 1] = end;
            */
        }
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
    add_transition(0, '[', 1);
    add_transition(1, '\\', 2);
    add_transition(1, ']', 0);
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

    cudaMemcpyToSymbol(d_D, D, NUM_STATES * NUM_CHARS * sizeof(int));
    cudaMemcpyToSymbol(d_E, E, NUM_STATES * NUM_CHARS * sizeof(uint8_t));

    int* h_output_array = new int[BUFFER_SIZE];

    std::ifstream is(INPUT_FILE);


    // get length of file:
    is.seekg (0, std::ios::end);
    long length = is.tellg();
    is.seekg (0, std::ios::beg);

    if(length > BUFFER_SIZE){
        cout<<"Error: File is too large to be read to buffer"<<endl;
    }
    else{
        string line; 
        long line_length;
        long line_count = 0; 
        long char_offset = 0; 

        // allocate memory:
        char* buffer = new char [BUFFER_SIZE];
        int* len_array = new int[NUM_LINES];
        int* offset_array = new int[NUM_LINES];

        while (getline(is, line)){

            line_length = line.size();

            // keep track of lengths of each line
            len_array[line_count] = line_length;

            // update offset from start of file
            char_offset += line_length;
            offset_array[line_count] = char_offset;

            // increment line index
            line_count++;

        }
        is.close();
        // reopen file stream
        std::ifstream is(INPUT_FILE);

        // read data as a block:
        is.read (buffer,length);

        //Memory allocation for kernel functions
    
        int* d_output_array;
        cudaMalloc((int**)&d_output_array, BUFFER_SIZE * sizeof(int));

        char* d_buffer;
        cudaMalloc((char**) &d_buffer, BUFFER_SIZE * sizeof(char));

        int* d_len_array;
        cudaMalloc((char**) &d_len_array, NUM_LINES * sizeof(int));

        int* d_offset_array;
        cudaMalloc((char**) &d_offset_array, NUM_LINES * sizeof(int));

        cudaMemcpy(d_buffer, buffer, BUFFER_SIZE * sizeof(char), cudaMemcpyHostToDevice);     
        cudaMemcpy(d_len_array, len_array, NUM_LINES * sizeof(int), cudaMemcpyHostToDevice);     
        cudaMemcpy(d_offset_array, offset_array, NUM_LINES * sizeof(int), cudaMemcpyHostToDevice);     

        dim3 dimGrid(NUM_LINES,1,1);
        dim3 dimBlock(NUM_THREADS,1,1);
        merge_scan<<<dimGrid, dimBlock>>>(1, d_buffer, d_len_array, BUFFER_SIZE, d_output_array);
       
        cudaMemcpy(h_output_array, d_output_array, BUFFER_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
        
        for(int j = 0; j < NUM_LINES; j++) {
            for(int i = 0; i < len_array[j]; i++) {
               //if(h_output_array[i + j * array_len] == 1) 
                   cout << i << " "; 
            }
            cout << endl;
        }

        clear_array<<<dimGrid, dimBlock>>>(d_output_array, BUFFER_SIZE);

        // close filestream
        is.close();

        // delete temporary buffer
        delete [] buffer;
        delete [] len_array;
        delete [] offset_array;

        cudaFree(d_output_array);
        cudaFree(d_buffer);
        cudaFree(d_len_array);
        cudaFree(d_offset_array);

        free(h_output_array);

    }


    return 0;
}


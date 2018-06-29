
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
#define NUM_THREADS 256
#define NUM_LINES 322
#define NUM_BLOCKS 30

#define BUFFER_SIZE 25000000
#define NUM_COMMAS 500
#define INPUT_FILE "./taxi_input.txt"

typedef std::chrono::high_resolution_clock Clock;

//Transition table for GPU function
__constant__ int     d_D[NUM_STATES * NUM_CHARS];
//Emission table for GPU function
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
        for(int i = 0; i < NUM_STATES; i++) 
            c.v[i] = b.v[a.v[i]];
        
        return c;
    }
};

__global__
void remove_empty_elements (int** input, int* len_array, int total_lines, int* index, int* temp_base, 
                            int* offset_array,  int* output, int* output_line_num) {

    __shared__ int line_num;
    __shared__ int base;

    int len;
    int block_num;


    if(threadIdx.x == 0) 
        line_num = atomicInc((unsigned int*) index, INT_MAX);
    __syncthreads();
    block_num =  line_num;

    

    while(block_num < total_lines) {

        len = len_array[block_num];


        if(threadIdx.x == 0)
            base = atomicAdd(temp_base, len);
        __syncthreads();
        

        offset_array[block_num] = (base);

        for(int loop = threadIdx.x; loop < len; loop += NUM_THREADS) {

            if(loop < len){
                output[base + loop] = (input[block_num])[loop];
                output_line_num[base + loop] = block_num;
            }
        }

        if(threadIdx.x == 0) {
            free(input[block_num]);
            line_num = atomicInc((unsigned int*) index, INT_MAX);
        }
         __syncthreads();
        block_num =  line_num;
    }

}


__global__
void merge_scan (char* line, int* len_array, int* offset_array, int** output_array, 
                 int* index, int total_lines, int* num_commas_array, SA* d_SA_Table, int* total_num_commas){


    typedef cub::BlockScan<SA, NUM_THREADS > BlockScan; // change name
    typedef cub::BlockScan<int, NUM_THREADS> BlockScan2; //

    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ typename BlockScan2::TempStorage temp_storage2;
    __shared__ SA prev_value;
    __shared__ int prev_sum;
    __shared__ int line_num;

    SA temp_prev_val;
    int temp_prev_sum;

    int len, offset;
    int block_num;
    int start_state;

    int* temp_output_array;
    int temp_array_size;

    if(threadIdx.x == 0) {
        line_num = atomicInc((unsigned int*) index, INT_MAX);
    }
    __syncthreads();
    block_num =  line_num;

    while(block_num < total_lines ) {

        temp_array_size = NUM_THREADS;
        //dynamic memory allocation
        if(threadIdx.x == 0) {
            temp_output_array = (int*)malloc(sizeof(int) * temp_array_size);
            output_array[block_num] = temp_output_array;
        }


        len = len_array[block_num];
        offset = offset_array[block_num];

        //initialize starting values
        SA a = SA();
        prev_value = a;
        temp_prev_val = SA();

        prev_sum = 0;
        temp_prev_sum = 0;
        int loop;

        //If the string is longer than NUM_THREADS
        for(int ph = 0; ph < len; ph += NUM_THREADS) {

            loop = threadIdx.x + ph;
            char c = 0;

            if(loop < len) {
                c = line[loop + offset ];
	            a = d_SA_Table[c];
            }
            __syncthreads();

            BlockScan(temp_storage).ExclusiveScan(a, a, prev_value, SA_op(), temp_prev_val);
            __syncthreads();
           
            start_state = prev_value.v[0];
            int state = a.v[start_state];
            int start = (int) d_E[(int) (NUM_CHARS * state + c)];
            int end;
            BlockScan2(temp_storage2).ExclusiveSum(start, end, temp_prev_sum);
            if(start == 1 && loop < len) {
                (output_array[block_num])[end + prev_sum] = loop;
            }

            if(threadIdx.x == 0) {
            	prev_value = temp_prev_val;
            	prev_sum += temp_prev_sum;
            }

            __syncthreads();

            if(threadIdx.x == 0) {
                if(prev_sum > (NUM_THREADS / 2)) {
                    temp_array_size += NUM_THREADS;
                    int* temp_ptr = (int*)malloc(sizeof(int) * temp_array_size);
                    for(int n = 0; n < prev_sum; n++) {
                        temp_ptr[n] = output_array[block_num][n];
                    }
                    free(output_array[block_num]);
                    output_array[block_num] = temp_ptr;
                }
            }
            __syncthreads();
                    
        }

        if(loop == len - 1) {
            num_commas_array[block_num] = prev_sum;
            int temp = atomicAdd(total_num_commas, prev_sum);
        }



        //to get the next line
        if(threadIdx.x == 0) 
            line_num = atomicInc((unsigned int*) index, INT_MAX);
         __syncthreads();
        block_num =  line_num;
    }


}




__global__
void polyline_coords (char* buffer, int* len_array, int* offset_array, int* comma_offset_array, int* comma_array,
                    int* output_len_array, int* output_offset_array, int total_lines){

        int loop = threadIdx.x + blockIdx.x * blockDim.x;
        if(loop < total_lines) {
            int offset = offset_array[loop];
            int len = len_array[loop];

            int start_idx = offset + comma_array[comma_offset_array[loop] + 7] + 3; 
            int end_idx = offset + len - 2;

            output_len_array[loop] = end_idx - start_idx;
            output_offset_array[loop] = start_idx;
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
SA 		SA_Table[NUM_CHARS];

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

void SA_generate () {
	for (int i = 0; i < NUM_CHARS; i++) {
		for(int j = 0; j < NUM_STATES; j++) {
			(SA_Table[i]).v[j] = D[j][i];
		}
	}
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

int main() {

    Dtable_generate();
    Etable_generate();
    SA_generate();

    SA* d_SA_Table;
    cudaMalloc((SA**) &d_SA_Table, NUM_CHARS * sizeof(SA));


    cudaMemcpyToSymbol(d_D, D, NUM_STATES * NUM_CHARS * sizeof(int));
    cudaMemcpyToSymbol(d_E, E, NUM_STATES * NUM_CHARS * sizeof(uint8_t));
    cudaMemcpy(d_SA_Table, SA_Table, NUM_CHARS * sizeof(SA), cudaMemcpyHostToDevice);



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
        int total_num_commas;

        // allocate memory:
        char* buffer = new char [BUFFER_SIZE];
        int* len_array = new int[NUM_LINES];
        int* offset_array = new int[NUM_LINES];
        int* comma_offset_array = new int[NUM_LINES];
        int* comma_len_array = new int [NUM_LINES];

        offset_array[0] = 0;

        while (getline(is, line)){

            line_length = line.size();

            // keep track of lengths of each line
            len_array[line_count] = line_length;

            // update offset from start of file
            char_offset += line_length + 1;
            offset_array[line_count + 1] = char_offset;

            // increment line index
            line_count++;

        }
        is.close();
        // reopen file stream
        std::ifstream is(INPUT_FILE);

        // read data as a block:
        is.read (buffer,length);

        // close filestream
        is.close();

        //Memory allocation for kernel functions
    
        int** d_output_array;
        cudaMalloc((int**)&d_output_array, line_count * sizeof(int*));

        char* d_buffer;
        cudaMalloc((char**) &d_buffer, BUFFER_SIZE * sizeof(char));

        int* d_len_array;
        cudaMalloc((int**) &d_len_array, line_count * sizeof(int));

        int* d_offset_array;
        cudaMalloc((int**) &d_offset_array, line_count * sizeof(int));

        int* d_num_commas;
        cudaMalloc((int**) &d_num_commas, line_count * sizeof(int));


        int* d_comma_offset_array;
        cudaMalloc((int**) &d_comma_offset_array, line_count * sizeof(int));


        int* d_stack;
        cudaMalloc((int**) &d_stack, sizeof(int));

        int* d_temp_base;
        cudaMalloc((int**) &d_temp_base, sizeof(int));

        int* d_total_num_commas;
        cudaMalloc((int**) &d_total_num_commas, sizeof(int));


        int temp = 0;

        auto t1 = Clock::now();

        cudaMemcpy(d_buffer, buffer, BUFFER_SIZE * sizeof(char), cudaMemcpyHostToDevice);     
        cudaMemcpy(d_len_array, len_array, line_count * sizeof(int), cudaMemcpyHostToDevice);     
        cudaMemcpy(d_offset_array, offset_array, line_count * sizeof(int), cudaMemcpyHostToDevice);    
        cudaMemcpy(d_stack, &temp, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_temp_base, &temp, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_total_num_commas, &temp, sizeof(int), cudaMemcpyHostToDevice);



        auto t2 = Clock::now();

        cout <<"Host to Device:" <<std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " microseconds" << endl;

        dim3 dimGrid(NUM_BLOCKS,1,1);
        dim3 dimBlock(NUM_THREADS,1,1);

        auto t3 = Clock::now();

        merge_scan<<<dimGrid, dimBlock>>>(d_buffer, d_len_array, d_offset_array, d_output_array, d_stack, line_count, d_num_commas, d_SA_Table, d_total_num_commas);

        cudaDeviceSynchronize();

        cudaMemcpy(&total_num_commas, d_total_num_commas, sizeof(int), cudaMemcpyDeviceToHost);

        int* d_final_array;
        cudaMalloc((int**) &d_final_array, total_num_commas * sizeof(int));


        int* d_line_idx_array;
        cudaMalloc((int**) &d_line_idx_array, total_num_commas * sizeof(int));

        int* h_output_array = new int[total_num_commas];

        cudaMemcpy(d_stack, &temp, sizeof(int), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();

        remove_empty_elements<<<dimGrid, dimBlock>>> (d_output_array, d_num_commas, line_count, d_stack, d_temp_base, d_comma_offset_array, d_final_array, d_line_idx_array);

        cudaDeviceSynchronize();

        auto t4 = Clock::now();
        cout << "data trans:" << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3).count() << " microseconds" << endl;


        auto t5 = Clock::now();
        //change the size later
        cudaMemcpy(h_output_array, d_final_array, total_num_commas * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(comma_len_array, d_num_commas, line_count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(comma_offset_array, d_comma_offset_array, line_count * sizeof(int), cudaMemcpyDeviceToHost);
        auto t6 = Clock::now();
        cout << "Device to Host:" << std::chrono::duration_cast<std::chrono::microseconds>(t6 - t5).count() << " microseconds" << endl;

        
        
        //  for(int i = 0; i < line_count; i++) {
        //     int len = comma_len_array[i];
           
        //     if(len >= 7) {
        //         int start = offset_array[i];
        //         int end = offset_array[i] + len_array[i];             

        //         int off = comma_offset_array[i]; 
        //         int comma_start = h_output_array[off + 7];
        //         for(int j = start + comma_start; j < end; j++) {
        //             cout << buffer[j];
        //         }

        //     }

        //     cout << endl;
        //     cout << endl;
        // }
        
        int* d_polyline_len_array;
        cudaMalloc((int**) &d_polyline_len_array, line_count * sizeof(int));

        int* d_polyline_offset_array;
        cudaMalloc((int**) &d_polyline_offset_array, line_count * sizeof(int));

        dim3 dimGridPoly(ceil(line_count/NUM_THREADS),1,1);

        polyline_coords<<<dimGrid, dimBlock>>>(d_buffer, d_len_array, d_offset_array, d_comma_offset_array, d_final_array, 
                d_polyline_len_array, d_polyline_offset_array, line_count);

        cudaDeviceSynchronize();

        cudaMemcpy(d_stack, &temp, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_temp_base, &temp, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_total_num_commas, &temp, sizeof(int), cudaMemcpyHostToDevice);

        int* d_polyline_num_commas;
        cudaMalloc((int**) &d_polyline_num_commas, line_count * sizeof(int));

        merge_scan<<<dimGrid, dimBlock>>>(d_buffer, d_polyline_len_array, d_polyline_offset_array, d_output_array, d_stack, line_count, d_polyline_num_commas, d_SA_Table, d_total_num_commas);

        cudaDeviceSynchronize();

        // int* polyline_len_array = new int[line_count];
        // int* polyline_offset_array = new int[line_count];

        // cudaMemcpy(polyline_len_array, d_polyline_len_array, sizeof(int) * line_count, cudaMemcpyDeviceToHost);
        // cudaMemcpy(polyline_offset_array, d_polyline_offset_array, sizeof(int) * line_count, cudaMemcpyDeviceToHost);

        // for(int i = 0; i < line_count; i++){
        //     printf("%.*s\n", polyline_len_array[i], buffer + polyline_offset_array[i]); 
        //     cout<<endl;
        //     cout<<endl;
        // }

        int polyline_total_num_commas;
        cudaMemcpy(&polyline_total_num_commas, d_total_num_commas, sizeof(int), cudaMemcpyDeviceToHost);

        int* d_polyline_array;
        cudaMalloc((int**) &d_polyline_array, polyline_total_num_commas * sizeof(int));

        int* p_output_array = new int[polyline_total_num_commas];

        cudaMemcpy(d_stack, &temp, sizeof(int), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();

        int* d_polyline_comma_offset_array;
        cudaMalloc((int**) &d_polyline_comma_offset_array, sizeof(int) * polyline_total_num_commas);


        int* d_line_num_array;
        cudaMalloc((int**) &d_line_num_array, sizeof(int) * polyline_total_num_commas);


        remove_empty_elements<<<dimGrid, dimBlock>>> (d_output_array, d_polyline_num_commas, line_count, d_stack, d_temp_base, d_polyline_comma_offset_array, d_polyline_array, d_line_num_array);

        cudaDeviceSynchronize();

        int* polyline_array = new int[polyline_total_num_commas];
        int* polyline_offset_array = new int[line_count];
        int* polyline_comma_len_array = new int [line_count];
        int* line_idx_array = new int[polyline_total_num_commas];
        int* polyline_comma_offset_array = new int[line_count];


        cudaMemcpy(polyline_array, d_polyline_array, sizeof(int) * polyline_total_num_commas, cudaMemcpyDeviceToHost);
        cudaMemcpy(polyline_comma_len_array, d_polyline_num_commas, sizeof(int) * line_count, cudaMemcpyDeviceToHost);
        cudaMemcpy(polyline_comma_offset_array, d_polyline_comma_offset_array, sizeof(int) * line_count, cudaMemcpyDeviceToHost);


        cudaMemcpy(polyline_offset_array, d_polyline_offset_array, sizeof(int) * line_count, cudaMemcpyDeviceToHost);

        cudaMemcpy(line_idx_array, d_line_num_array, sizeof(int) * polyline_total_num_commas, cudaMemcpyDeviceToHost);


        // for(int i = 0; i < polyline_total_num_commas; i++){
        //     int start = polyline_offset_array[line_idx_array[i]] + polyline_array[polyline_offset_array];
        //     int end =
        //     cout << polyline_array[i] << endl;
        // }

        for(int i = 0; i < line_count; i ++) {
            int num = polyline_comma_len_array[i];
            int comma_off2 = polyline_comma_offset_array[i];
            for(int j = 0; j <= num; j++) {
                if((j != num) && (j != 0)) {

                    printf("%.*s\n", polyline_array[comma_off2 + j] - polyline_array[j - 1 + comma_off2] - 2, buffer + polyline_array[j - 1 + comma_off2] + polyline_offset_array[i] + 2);
                }

                else if(j == 0) {

                    printf("%.*s\n", polyline_array[j + comma_off2], buffer + polyline_offset_array[i]);
                }

                else{

                    int comma_off = comma_offset_array[i];
                    int first_comma_idx = h_output_array[comma_off];
                    printf("%.*s\n", len_array[i] - (polyline_offset_array[i] - offset_array[i]) - polyline_array[j - 1 + comma_off2] - 4, buffer + polyline_array[j - 1 + comma_off2] + polyline_offset_array[i] + 2);
                }
            }
            cout<<endl;
        }


        cudaFree(d_polyline_len_array);
        cudaFree(d_polyline_offset_array);
        cudaFree(d_output_array);
        cudaFree(d_buffer);
        cudaFree(d_len_array);
        cudaFree(d_offset_array);
        cudaFree(d_comma_offset_array);
        cudaFree(d_stack);
        cudaFree(d_temp_base);
        cudaFree(d_num_commas);

        cudaFree(d_line_num_array);
        cudaFree(d_line_idx_array);


        // delete temporary buffers
        delete [] buffer;
        delete [] len_array;
        delete [] offset_array;
        delete [] comma_offset_array;
        delete [] comma_len_array;
        delete [] h_output_array;

        delete [] line_idx_array;

    }



    return 0;
}



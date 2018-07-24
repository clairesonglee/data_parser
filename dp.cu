
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
#define NUM_THREADS 352
#define NUM_LINES 51000
#define NUM_BLOCKS 20

#define BUFFER_SIZE 250000000 //in byte
//#define INPUT_FILE "./input/go_track_trackspoints.csv"
//#define INPUT_FILE "./input/gtt_double.csv"
//#define INPUT_FILE "./input/sfpd_plots_400.csv"
//#define INPUT_FILE "./input/flight_1_rgb_1.csv.csv"
//#define INPUT_FILE "./input/bacteria_4.csv"
//#define INPUT_FILE "./input/nla.csv"
//#define INPUT_FILE "./input/plot2.csv"



#define CSV_FILE 1 // 1: csv file, 0: txt file


typedef std::chrono::high_resolution_clock Clock;

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

/*
    remove_empty_elements:

    This function transfers the data from the pre-allocated array into the correct sized output array in order to 
    remove all internal fragmentations.
*/

__global__
void remove_empty_elements (int** input, int* len_array, int** off_array, int total_lines, long* index, long* temp_base, 
                            int* offset_array,  int* output, int* output_line_num, int* array_index, int taxi_application) {

    __shared__ int s_line_num;
    __shared__ int base;

    int len;
    int* off;
    int line_num;

    //get the next line to compute
    if(threadIdx.x == 0) 
        s_line_num = atomicInc((unsigned int*) index, INT_MAX);
    __syncthreads();
    // all threads have the same line_num
    line_num =  s_line_num;

        //printf("line_num: %d\n", line_num);

    while(line_num < total_lines) {

        //get the length of the line
        len = len_array[line_num];
        off = off_array[line_num];


        //get the offset in order to save the output into the correct place
        if(threadIdx.x == 0)
            base = offset_array[line_num];
        __syncthreads();
        
        // for loop for the line that is longer than NUM_THREADS(number of threads)
        for(int loop = threadIdx.x; loop < len; loop += NUM_THREADS) {

                //if the current character is in the line, it copies the value into the output array
                if(loop < len){
                   //  output[base + loop] = (input[line_num])[loop];
                   output[base + loop] = off[loop];
                   //  printf("%d\n", input[line_num][loop]);
                }
            

        }
        __syncthreads();

        if(threadIdx.x == 0) {
    
            s_line_num = atomicInc((unsigned int*) index, INT_MAX);
        }
         __syncthreads();
         //update the line_num for all threads
        line_num =  s_line_num;
    }

    // if(threadIdx.x + blockIdx.x == 0) {
    //     for(int i = 0; i < array_index[0]; i++) {
    //         free(input[i]);
    //     }
    // }
     int num_a = array_index[0];
    for(int i = 0; i < (int)ceilf((float) (num_a) / (NUM_BLOCKS * NUM_THREADS)); i++) {
        int loop = threadIdx.x + blockIdx.x * blockDim.x + i * NUM_BLOCKS * NUM_THREADS;
        if(loop < num_a) {
            free(input[loop]);
        }
    }

}


__global__
void merge_scan (char* line, int* len_array, int* offset_array, int** output_array, 
                 long* index, int* array_index, int total_lines, int* num_commas_array, int** off_commas_array, SA* d_SA_Table, int* total_num_commas, uint8_t* d_E, int taxi_application){

    typedef cub::BlockScan<SA, NUM_THREADS> BlockScan_exclusive_scan; // change name
    typedef cub::BlockScan<int, NUM_THREADS> BlockScan_exclusive_sum; //

    __shared__ typename BlockScan_exclusive_scan::TempStorage temp_storage;
    __shared__ typename BlockScan_exclusive_sum::TempStorage temp_storage2;
    __shared__ SA prev_value;
    __shared__ int prev_sum;
    __shared__ int s_line_num;
    __shared__ int s_array_num;
    __shared__ int s_mem_space;
    __shared__ int* s_current_array;
    __shared__ int stack;


    //clock_t c_start = clock();

    SA temp_prev_val;
    int temp_prev_sum;

    int len, offset;
    int line_num;
    int start_state;

    int* temp_output_array;
    int temp_array_size;

    int array_num;
    //int stack ;

    //grab the next/new line to compute
    if(threadIdx.x == 0) {
        s_line_num = atomicInc((unsigned int*) index, INT_MAX);
        s_array_num = atomicInc((unsigned int*) array_index, INT_MAX);
    }

    __syncthreads();
    line_num = s_line_num;
    array_num = s_array_num;

    temp_array_size = total_lines * 20 / NUM_BLOCKS;
    //temp_array_size = 100;

    if(threadIdx.x == 0) {
        temp_output_array = (int*)malloc(sizeof(int) * temp_array_size);
        output_array[array_num] = temp_output_array;
        s_current_array = output_array[array_num];
        s_mem_space = temp_array_size;
        stack = 0;
    }
    __syncthreads();


    // long clock_temp = 0;
    // long clock_scan = 0;
    // long clock_end = 0;
    // long clock_malloc = 0;
    // long clock_copy = 0;

    //if the current line is in the input file 
    while(line_num < total_lines ) {
        // clock_t c1 = clock();
        len = len_array[line_num];
        offset = offset_array[line_num];

        //initialize starting values
        SA a;
        prev_value = a;

        prev_sum = 0;
        temp_prev_sum = 0;
        int loop;
        __syncthreads();

        // clock_t c2 = clock();
        // clock_temp += (c2 - c1);
        

        //If the string is longer than NUM_THREADS
        for(int ph = 0; ph < len; ph += NUM_THREADS) {

           // clock_t c3 = clock();

            loop = threadIdx.x + ph;
            char c = 0;

            if(loop < len) {
                c = line[loop + offset ];
                a = d_SA_Table[c];
            }
            start_state = prev_value.v[0];

            __syncthreads();
            //Merge SAs (merge the data to make one final sequence)
            BlockScan_exclusive_scan(temp_storage).ExclusiveScan(a, a, prev_value, SA_op(), temp_prev_val);
            __syncthreads();
           
            int state = a.v[start_state];
            int start = (int) d_E[(int) (NUM_CHARS * state + c)];
            int end;
            //Excusive sum operation to find the number of commas
            BlockScan_exclusive_sum(temp_storage2).ExclusiveSum(start, end, temp_prev_sum);

            //(if the array is full, then it doubles the size fo the array )

            if(prev_sum + temp_prev_sum > s_mem_space){
                printf("new array\n");
                 if(threadIdx.x == 0){        
                     s_array_num = atomicInc((unsigned int*) array_index, INT_MAX);
                     temp_output_array = (int*)malloc(sizeof(int) * temp_array_size);
                     output_array[s_array_num] = temp_output_array;
                     s_current_array = output_array[s_array_num];
                     s_mem_space = temp_array_size;
                     stack = 0;
                  }
                  __syncthreads();
                  array_num = s_array_num;
            }

            __syncthreads();

            //save the data (comma_index)
            if(start == 1 && loop < len) {
                (output_array[array_num])[end + prev_sum + stack] = loop;
                // if(array_num == 0) {
                //     printf("end %d, stack: %d loop: %d\n", end + prev_sum, stack, loop);
                // }
            }

            //save the end values for the next iteration
            if(threadIdx.x == NUM_THREADS - 1) {
                //prev_value = temp_prev_val;
                prev_value = a;

                prev_sum += temp_prev_sum;
            }
            __syncthreads();

            // clock_t c4 = clock();
            // clock_scan += c4 - c3;   
                    
        }
      // clock_t c5 = clock();

        //if the last thread saves the number of commas in the line
        if(loop == len - 1) {
            s_mem_space -= prev_sum;
            num_commas_array[line_num] = prev_sum;
            off_commas_array[line_num] = s_current_array;
            if(prev_sum > 0){
                stack += prev_sum;
            }

            s_current_array = (int*) (s_current_array + prev_sum);
            //atomic operation to track total number of commas in the input file to properly allocated the space for the output array
            int temp = atomicAdd(total_num_commas, prev_sum);
        }

        //to get the next line
        if(threadIdx.x == 0) 
            s_line_num = atomicInc((unsigned int*) index, INT_MAX);
         __syncthreads();
        line_num =  s_line_num;
        // clock_t c6 = clock();
        //     clock_end += c6 - c5;   
    }
    // clock_t c_end = clock();
    // if(threadIdx.x + blockIdx.x == 0){
    //     printf("merge_scan tiem: %d\n", c_end - c_start);
    //     // printf("setup time: %d\n", clock_temp);
    //     // printf("malloc time: %d\n", clock_malloc);
    //     printf("copy time: %d\n", clock_copy);

    //     printf("scan time: %d\n", clock_scan);
    //     printf("setup time2: %d\n", clock_end);
    // }

   // __syncthreads();
    // if(threadIdx.x + blockIdx.x == 0) {
    //     for(int i = 0; i < temp_array_size; i++){
    //         printf("%d ",output_array[0][i]);
    //     }
    // }
}

/*
    This is just a function that calles ExclusiveSum function.
*/

__global__
void output_sort(int* input, int len, int* output) {
    typedef cub::BlockScan<int, NUM_THREADS> BlockScan; 
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ int prev_sum;

    int temp_prev_sum = 0;
    prev_sum = 0;

    //if the len is longer than the number of threads
    for(int ph = 0; ph < (int)ceilf((float) (len) / NUM_THREADS); ph ++) {
        int loop = threadIdx.x + ph * NUM_THREADS;
        temp_prev_sum = prev_sum;

        int start = input[loop];
        int end;
        BlockScan(temp_storage).ExclusiveSum(start, end, temp_prev_sum);
        
        if(loop < len)
            output[loop] = end + prev_sum;
        __syncthreads();
        //save the last value as well (the output array has one more element than the input array because the first element is 0)
        if(loop == len - 1)
            output[loop + 1] = temp_prev_sum + prev_sum;
        __syncthreads();
        
        if (threadIdx.x == 0)
            prev_sum += temp_prev_sum;
        __syncthreads();

    }

    __syncthreads();

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
SA      SA_Table[NUM_CHARS];

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
   // add_default_transition(3 , 0);

    add_transition(0, '{', 1);
    add_transition(1, '\\', 2);
    add_transition(1, '}', 0);
 //   add_transition(0, '\\', 3);
}

void Etable_generate() 
{
    for(int i = 0; i < NUM_STATES; i++) 
        add_default_emission(i, 0);
    
    add_emission(0, ',', 1);
}

int main() {


    string INPUT_FILE = "./input/nla.csv";

    // generate transition tables for constant lookup time
    Dtable_generate();
    Etable_generate();
    SA_generate();

    // allocate device memory for state tables
    SA* d_SA_Table;
    cudaMalloc((SA**) &d_SA_Table, NUM_CHARS * sizeof(SA));
    cudaDeviceSynchronize();

    auto t10 = Clock::now();       

    uint8_t* d_E;
    cudaMalloc((uint8_t**) &d_E, NUM_STATES * NUM_CHARS * sizeof(uint8_t));
    auto t11 = Clock::now();
    cout <<"Seq setup:" <<std::chrono::duration_cast<std::chrono::microseconds>(t11 - t10).count() << " microseconds" << endl;


    // copy state tables from host to device
    cudaMemcpy(d_E, E, NUM_STATES * NUM_CHARS * sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_SA_Table, SA_Table, NUM_CHARS * sizeof(SA), cudaMemcpyHostToDevice);


    // open input file
    std::ifstream is(INPUT_FILE);

    // get length of file:
    is.seekg (0, std::ios::end);
    long length = is.tellg();
    is.seekg (0, std::ios::beg);


    // if length of file is greater than buffer, send error message
    if(length > BUFFER_SIZE){
        cout<<"Error: File is too large to be read to buffer"<<endl;
    }
    else{
        string line; 
        long line_length;       // holds length of line being read
        long line_count = 0;    // counts number of lines read
        long char_offset = 0;   // offset of line into buffer
        int total_num_commas;   // number of commas 

        // allocate memory for arrays 
        char* buffer = new char [BUFFER_SIZE];
        int* len_array = new int[NUM_LINES];
        int* offset_array = new int[NUM_LINES + 1];
       
        // initialize offset array
        offset_array[0] = 0;   
        // read line by line from input file 
        while (getline(is, line)){
            // keep track of line length 
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

        auto tt1 = Clock::now();

        //Memory allocation for kernel functions

        int buffer_len = offset_array[line_count];
      //  printf("line_count: %d\n", line_count);

        int** d_output_array;
        cudaMalloc((int**)&d_output_array, line_count * sizeof(int*));

        char* d_buffer;
        cudaMalloc((char**) &d_buffer, buffer_len * sizeof(char));

        int* d_len_array;
        cudaMalloc((int**) &d_len_array, line_count * sizeof(int));

        int* d_offset_array;
        cudaMalloc((int**) &d_offset_array, line_count * sizeof(int));


        int* d_num_commas;
        cudaMalloc((int**) &d_num_commas, line_count * sizeof(int));

        int** d_off_commas;
        cudaMalloc((int**) &d_off_commas, line_count * sizeof(int*));


        int* d_comma_offset_array;
        cudaMalloc((int**)&d_comma_offset_array, (line_count + 1) * sizeof(int));

        long* d_stack;
        cudaMalloc((long**) &d_stack, sizeof(long));

        long* d_temp_base;
        cudaMalloc((long**) &d_temp_base, sizeof(long));

        int* d_total_num_commas;
        cudaMalloc((int**) &d_total_num_commas, sizeof(int));

        int* d_array_index;
        cudaMalloc((int**) &d_array_index, sizeof(int));

        cudaDeviceSynchronize();
        auto tt2 = Clock::now();

       // cout <<"Device M A:" <<std::chrono::duration_cast<std::chrono::microseconds>(tt2 - tt1).count() << " microseconds" << endl;

        long temp = 0;
        

        
        auto t1 = Clock::now();

        // copies host memory to device memory after allocation 
        cudaMemcpy(d_buffer, buffer, buffer_len * sizeof(char), cudaMemcpyHostToDevice);     
        cudaMemcpy(d_len_array, len_array, line_count * sizeof(int), cudaMemcpyHostToDevice);     
        cudaMemcpy(d_offset_array, offset_array, line_count * sizeof(int), cudaMemcpyHostToDevice);    
        cudaMemcpy(d_stack, &temp, sizeof(long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_temp_base, &temp, sizeof(long), cudaMemcpyHostToDevice);
        cudaMemcpy(d_total_num_commas, &temp, sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_array_index, &temp, sizeof(int), cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        auto t2 = Clock::now();

      //  cout <<"Host to Device:" <<std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << " microseconds" << endl;

        // grid and block dimensions to launch kernel function
        dim3 dimGrid(NUM_BLOCKS,1,1);
        dim3 dimBlock(NUM_THREADS,1,1);

        int nBlocksPerSM_suggested;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nBlocksPerSM_suggested, merge_scan, NUM_THREADS, 4432);
       // printf("nblock: %d\n", nBlocksPerSM_suggested);

        //dim3 dimGrid_Merge(nBlocksPerSM_suggested * 100, 1, 1);
        
        //t3 ~ t4 for merge_can and output sort
        size_t limit = 0;
        cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
        printf("cudaLimitMallocHeapSize: %u\n", (unsigned)limit);
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, limit * 100);

        auto t3 = Clock::now();

//         cudaDeviceSynchronize();
//         cudaError_t e = cudaGetLastError();
//         if(e!=cudaSuccess) {                                              
//             printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           
//         } 
        // function call to locate commas in input line
        merge_scan<<<dimGrid, dimBlock>>>(d_buffer, d_len_array, d_offset_array, d_output_array, d_stack, d_array_index, line_count, d_num_commas, d_off_commas, d_SA_Table, d_total_num_commas, d_E, 0);
        


        //offset array has the correct offset values
        output_sort<<<1, NUM_THREADS>>> (d_num_commas, line_count, d_comma_offset_array);


        // copies number of commas to host memory
        cudaDeviceSynchronize();
        auto t4 = Clock::now();


        cudaMemcpy(&total_num_commas, d_total_num_commas, sizeof(int), cudaMemcpyDeviceToHost);




        // creates an array to hold commas in device memory
        int* d_final_array;
        cudaMalloc((int**) &d_final_array, total_num_commas * sizeof(int));
      
        // makes a temporary d_stack
        cudaMemcpy(d_stack, &temp, sizeof(int), cudaMemcpyHostToDevice);
       
        auto t5 = Clock::now();

        // launches kernel function to clear unnecessary information
        remove_empty_elements<<<dimGrid, dimBlock>>> (d_output_array, d_num_commas, d_off_commas, line_count, d_stack, d_temp_base, d_comma_offset_array, d_final_array, d_final_array /* temp array */, d_array_index, 0);
        

        cudaDeviceSynchronize();
        auto t6 = Clock::now();

        cout << "data trans:" << std::chrono::duration_cast<std::chrono::microseconds>(t4 - t3 + t6 - t5).count() << " microseconds" << endl;

        int* num_commas = new int[line_count];
        int* comma_offset_array = new int[line_count + 1];
        int* final_array = new int[total_num_commas];

        auto t7 = Clock::now();
        cudaMemcpy(num_commas, d_num_commas, line_count * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(comma_offset_array, d_comma_offset_array, (line_count + 1) * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(final_array, d_final_array,  total_num_commas * sizeof(int), cudaMemcpyDeviceToHost);     

        cudaDeviceSynchronize();
        auto t8 = Clock::now();
        cout << "Device to Host:" << std::chrono::duration_cast<std::chrono::microseconds>(t8 - t7).count() << " microseconds" << endl;

        // for(int i = 0; i < line_count; i++){
        //     int len = num_commas[i];
        //     int off = comma_offset_array[i];
        //     for(int j = 0; j < len; j++){
        //         printf("%d ", final_array[off + j]);
        //     }
        //     printf("\n");
        // }

     
        // device memory


        cudaFree(d_output_array);
        cudaFree(d_buffer);
        cudaFree(d_len_array);
        cudaFree(d_offset_array);
        cudaFree(d_comma_offset_array);

        cudaFree(d_stack);
        cudaFree(d_temp_base);
        cudaFree(d_num_commas);

        cudaFree(d_total_num_commas);

        cudaFree(d_SA_Table);
        cudaFree(d_E);
        cudaFree(d_final_array);

        // delete temporary buffers
        delete [] buffer;
        delete [] len_array;
        delete [] offset_array;

        delete [] final_array;
        delete [] num_commas;
        delete [] comma_offset_array;

    }


    return 0;
}



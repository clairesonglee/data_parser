#!/bin/sh
"""
# change params 

sed -i '' 's+^#define INPUT_FILE .*$+#define INPUT_FILE '$1'+' test.cpp

sed -i '' 's/^#define NUM_STATES .*$/#define NUM_STATES '$2'/' test.cpp

sed -i '' 's/^#define NUM_THREADS .*$/#define NUM_THREADS '$3'/' test.cpp


#sed -i 's/^#define INPUT_FILE .*$/#define INPUT_FILE '"testing.csv"'/' test.cpp

#compile and run exe 
g++ -o test test.cpp && ./test



"""

# change params 
sed -i 's+^#define INPUT_FILE .*$+#define INPUT_FILE '$1'+' test.cpp

sed -i 's/^#define NUM_STATES .*$/#define NUM_STATES '$1'/' dp_auto.cu

sed -i 's/^#define NUM_THREADS .*$/#define NUM_THREADS '$2'/' dp_auto.cu

declare -i output
output=$(expr "$2")
if [ $output -gt 256 ];then
    if [[ $output -lt 385 ]];then
 	nvcc -arch=compute_52 -code=sm_52 -o dp_auto dp_auto.cu -I/export/project/hondius/opt/cub -std=c++11 -maxrregcount=168 && ./dp_auto
    else
	nvcc -arch=compute_52 -code=sm_52 -o dp_auto dp_auto.cu -I/export/project/hondius/opt/cub -std=c++11 -maxrregcount=128 && ./dp_auto
    fi
else
    nvcc -arch=compute_52 -code=sm_52 -o dp_auto dp_auto.cu -I/export/project/hondius/opt/cub -std=c++11 && ./dp_auto
fi

# print output.txt contents 
cat output.csv >> out.csv 

#"""

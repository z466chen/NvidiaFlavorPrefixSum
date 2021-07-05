#include <iostream>
#include <chrono>

#define BLOCK_SIZE 1024
#define NUM_OF_BANKS 32
#define LOG_NUM_OF_BANKS 5
#define SHIFT_BANK(n) \
    (n + (n >> LOG_NUM_OF_BANKS))


__global__ void prefix_sum(float *in, float *out, float* aux, int noc, int res) {
    __shared__ float temp[2*BLOCK_SIZE];
    int n = BLOCK_SIZE*2;
    if (blockIdx.x == noc - 1) n = res;

    int thid = threadIdx.x;
    // printf("thid: %d, bid: %d, noc: %d\n", thid, blockIdx.x, noc);
    temp[SHIFT_BANK(2*thid)] = in[2*(thid + blockIdx.x*BLOCK_SIZE)];
    temp[SHIFT_BANK(2*thid+1)] = in[2*(thid + blockIdx.x*BLOCK_SIZE)+1];
    
    int offset = 1;
    for (int d = (n >> 1); d > 0; d >>= 1) {
        __syncthreads();
        if (thid < d) {
            int ai = SHIFT_BANK(offset*(2*thid+1) - 1);
            int bi = SHIFT_BANK(offset*(2*thid+2) - 1);
            temp[bi] += temp[ai];
        }
        offset <<= 1;
    }
    
    __syncthreads();
    
    
    if (thid == 0) {
        aux[blockIdx.x] = temp[SHIFT_BANK(n - 1)];
        temp[SHIFT_BANK(n-1)] = 0;
    }

    for (int d = 1; d <= (n >> 1); d <<= 1) {
        offset >>= 1;
        __syncthreads();
        if (thid < d) {
            int ai = SHIFT_BANK(offset*(2*thid+1) - 1);
            int bi = SHIFT_BANK(offset*(2*thid+2) - 1);
            float t = temp[ai] + temp[bi];
            temp[ai] = temp[bi];
            temp[bi] = t;
        }
    }

    
    
    __syncthreads();
    out[2*(threadIdx.x + blockIdx.x * BLOCK_SIZE)] = temp[SHIFT_BANK(2*thid)];
    out[2*(threadIdx.x + blockIdx.x * BLOCK_SIZE) + 1] = temp[SHIFT_BANK(2*thid+1)];
}

__global__ void block_add(float *in, float *out){
    out[2*(threadIdx.x + blockIdx.x * BLOCK_SIZE)] += in[blockIdx.x];
    out[2*(threadIdx.x + blockIdx.x * BLOCK_SIZE) + 1] += in[blockIdx.x];
}

void prefix_sum_cpu_rec(float *in, float *out, int noc, int n) {
    float *aux;
    cudaMalloc((void **)&aux, noc*sizeof(float));
    int res = 2*BLOCK_SIZE;
    if (n%(2*BLOCK_SIZE) != 0 || n == 0) res = n%(2*BLOCK_SIZE); 
    prefix_sum<<<noc, BLOCK_SIZE>>>(in, out, aux, noc, res);
    
    cudaDeviceSynchronize();
    if (noc == 1) {
        cudaFree(aux);
        return;
    }
    float *auxout;
    cudaMalloc((void **)&auxout, ((noc + 2*BLOCK_SIZE - 1)/(2*BLOCK_SIZE))*sizeof(float));
    prefix_sum_cpu_rec(aux, auxout, (noc + 2*BLOCK_SIZE - 1)/(2*BLOCK_SIZE), noc);
    block_add<<<noc, BLOCK_SIZE>>>(auxout, out);
    cudaDeviceSynchronize();
}

int main() {
    int n = 1 << 20;
    float *in, *out;

    int noc = (n + 2*BLOCK_SIZE - 1)/(2*BLOCK_SIZE);
    cudaMalloc((void **)&in, n*sizeof(float));
    cudaMalloc((void **)&out, n*sizeof(float));
    
    srand(0);    

    float test[n];
    for (int i = 0; i < n; ++i) {
        float t = (rand()%10)/10.0f;
        test[i] = t;  
    }

    cudaMemcpy(in, test, n*sizeof(float), cudaMemcpyHostToDevice);
    
    auto start = std::chrono::steady_clock::now();

    prefix_sum_cpu_rec(in,out, noc, n);    

    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    std::cout << std::chrono::duration <double, std::milli> (diff).count() << " ms" << std::endl;
    printf("%s\n", cudaGetErrorString(cudaGetLastError()));
    for (int i = n - 1; i < n; ++i) {
        float a;
        cudaMemcpy(&a, out+i, sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << a << std::endl;
    }
    
    cudaFree(in);
    cudaFree(out);
}
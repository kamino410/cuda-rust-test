extern "C" {
    __global__ void vectorAdd(const float *a, const float *b, float *c, int num) {
        int i = blockDim.x * blockIdx.x + threadIdx.x;
        if (i < num) {
            c[i] = a[i] + b[i];
        }
    }

    __global__ void initImage(unsigned char *data, int cols, int rows) {
        int x = threadIdx.x + blockIdx.x * blockDim.x;
        int y = threadIdx.y + blockIdx.y * blockDim.y;
        if (x >= cols || y >= rows) return;

        int offset = 3 * (x + y * cols);

        data[offset] = 255 * ((float)x / cols);
        data[offset + 1] = 255 * ((float)y / rows);
        data[offset + 2] = 128;
    }
}

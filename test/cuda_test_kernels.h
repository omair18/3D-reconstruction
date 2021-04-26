
void print_elements_api(int blocks, int threads, int elementsCount, unsigned char* devPtr);

void elementwise_divide_float_api(float divider, float* gpuData, size_t width, size_t height, size_t pitch, size_t channels, void* cudaStream);

void compute_gradients_api(float* dataLx, float* dataLy, float* resultData, size_t width, size_t height, size_t pitch, size_t channels, void* cudaStream);

void preprocess_histogram_api(float* data, size_t width, size_t height, size_t pitch, int nbins, float maxGradient, void* cudaStream);
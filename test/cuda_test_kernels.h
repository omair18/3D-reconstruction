
void print_elements_api(int blocks, int threads, int elementsCount, unsigned char* devPtr);

void compute_gradients_api(float* dataLx, float* dataLy, float* resultData, size_t width, size_t height, size_t pitch, size_t channels, void* cudaStream);

void preprocess_histogram_api(float* data, size_t width, size_t height, size_t pitch, int nbins, float maxGradient, void* cudaStream);

void linear_sampling_api(float* in, float* out, size_t in_width, size_t in_height, size_t in_pitch, size_t out_width, size_t out_height, size_t out_pitch);

void perona_malik_g1_diffusion_api(float* Lx_data, size_t Lx_pitch, float* Ly_data, size_t Ly_pitch, float contrastFactor, float* out_data, size_t out_width, size_t out_height, size_t out_pitch);

void perona_malik_g2_diffusion_api(float* Lx_data, size_t Lx_pitch, float* Ly_data, size_t Ly_pitch, float contrastFactor, float* out_data, size_t out_width, size_t out_height, size_t out_pitch);

void weickert_diffusion_api(float* Lx_data, size_t Lx_pitch, float* Ly_data, size_t Ly_pitch, float contrastFactor, float* out_data, size_t out_width, size_t out_height, size_t out_pitch);

void charbonier_diffusion_api(float* Lx_data, size_t Lx_pitch, float* Ly_data, size_t Ly_pitch, float contrastFactor, float* out_data, size_t out_width, size_t out_height, size_t out_pitch);

void fast_explicit_diffusion_api(float* src, float* diff, float* temp, size_t width, size_t height, size_t srcPitch, size_t diffPitch, size_t tempPitch, );
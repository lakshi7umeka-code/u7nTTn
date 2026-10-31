#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <cuda.h>

namespace transfAot{
class KwD;

class Trainer{
    private:
        float* d_targetPtr = nullptr;
        cudaStream_t train_stream=nullptr;
        cudaStream_t execution_stream=nullptr;
        
    public:
        Trainer(KwD* model_weights,cudaStream_t train_stream);
       
        void forward();
        void backwardCompatibility(cudaStream_t student_stream);
        void backward(cudaStream_t student_stream);
};
class ATTN{
    private:
        CUmodule attn_module;
        CUfunction attn_kernel;
    public:
        ATTN();
        ~ATTN();
        float* forwardAttn();
        float* backwardAttn();
};
class FFN{
    private:
        CUmodule ffn_module;
        CUfunction ffn_kernel;
    public:
        FFN();
        ~FFN();
        float* forwardFFn();
        float* backwardFFn();
};
class Encoder{
    private:
        std::unique_ptr<FFN> ffn_layer;
        std::unique_ptr<ATTN> attn_layer;

        cudaStream_t ffn_stream, attn_stream;
    public:
        Encoder();
        ~Encoder();
        float* forwardEnc(float* d_in,int b,int h,int s);
        float* backwardEnc();
};
class KwD{
    private:
        std::unique_ptr<Trainer> student, teacher;
        std::unique_ptr<Encoder> encoder_layer;

        cudaStream_t student_stream, teacher_stream, memOps_stream; 

        size_t m_weightBytes = 0;float* device_weights=nullptr;

        cudaMemPool_t model_memPool;
        cudaEvent_t device_weights_ready;
    public:
        KwD();
        ~KwD();
        void load_model(const std::string& model_path, size_t totalElements);
        float* getDevicePointer() const{return device_weights;};
        cudaEvent_t getReadyEvent() const{return device_weights_ready;};
};
}

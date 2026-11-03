#include <iostream>
#include <memory>
#include <string>
#include <cuda_runtime.h>
#include <cuda.h>

namespace transfAot{
class KwD;

class ATTN{
    private:
        CUmodule attn_module;
        CUfunction attn_kernel;
    public:
        ATTN();
        ~ATTN();
        float* forwardAttn(float* d_in,
                           float* model_weights,
                           int batch,
                           int hidden,
                           int sequence,
                           cudaStream_t attn_stream);
        float* backwardAttn(float* output_gradient,
                            float* model_weights,
                            int batch,
                            int hidden,
                            int sequence,
                            cudaStream_t attn_stream);
};
class FFN{
    private:
        CUmodule ffn_module;
        CUfunction ffn_kernel;
    public:
        FFN();
        ~FFN();
        float* forwardFFn(float* attention_output,
                          float* model_weights,
                          int batch,
                          int hidden,
                          int sequence,
                          cudaStream_t ffn_stream);
        float* backwardFFn(float* output_gradient,
                           float* model_weights,
                           int batch,
                           int hidden,
                           int sequence,
                           cudaStream_t ffn_stream);
};
class Encoder{
    private:
        std::unique_ptr<FFN> ffn_layer;
        std::unique_ptr<ATTN> attn_layer;

        cudaStream_t ffn_stream, attn_stream;
    public:
        Encoder();
        ~Encoder();
        float* forwardEnc(float* d_in, float* model_weights, int batch, int hidden, int sequence);
        float* backwardEnc(float* output_gradient,
                           float* model_weights,
                           int batch,
                           int hidden,
                           int sequence);
};
class Trainer{
    private:
        friend class KwD;

        float* d_modelWeights = nullptr;
        cudaStream_t train_stream=nullptr;
        cudaStream_t execution_stream=nullptr;
        bool trainable=false;
        
        std::unique_ptr<Encoder> encoder_layer;

        float* forward(float* input, int batch, int hidden, int sequence);
        float* backwardCompatibility(cudaStream_t requested_stream,
                                     float* output_gradient,
                                     int batch,
                                     int hidden,
                                     int sequence);
    public:
        Trainer(float* model_weights,
                cudaEvent_t weights_ready,
                cudaStream_t train_stream,
                bool trainable);
};
class KwD{
    private:
        std::unique_ptr<Trainer> student, teacher;
        

        cudaStream_t student_stream, teacher_stream, memOps_stream; 

        size_t m_weightBytes = 0;
        float* student_weights=nullptr;
        float* teacher_weights=nullptr;

        cudaMemPool_t model_memPool=nullptr;
        cudaEvent_t device_weights_ready=nullptr;
    public:
        KwD();
        ~KwD();
        void load_model(const std::string& model_path, size_t totalElements);
        float* teacherForward(float* input, int batch, int hidden, int sequence);
        float* studentForward(float* input, int batch, int hidden, int sequence);
        float* studentBackward(float* output_gradient, int batch, int hidden, int sequence);
        float* getStudentDevicePointer() const{return student_weights;};
        float* getTeacherDevicePointer() const{return teacher_weights;};
        cudaEvent_t getReadyEvent() const{return device_weights_ready;};
};
}

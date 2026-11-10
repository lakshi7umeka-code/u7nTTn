#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <vector>
#include "orch.hpp"

namespace transfAot {
namespace {
void checkCuda(cudaError_t status, const char* operation){
    if(status != cudaSuccess){
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

void checkCu(CUresult status, const char* operation){
    if(status != CUDA_SUCCESS){
        const char* error_text = nullptr;
        cuGetErrorString(status, &error_text);
        throw std::runtime_error(std::string(operation) + ": " +
                                 (error_text != nullptr ? error_text : "CUDA Driver API error"));
    }
}
}

Trainer::Trainer(float* model_weights,
                 cudaEvent_t weights_ready,
                 cudaStream_t train_stream,
                 bool trainable)
    : d_modelWeights(model_weights),
      train_stream(train_stream),
      trainable(trainable){
    encoder_layer=std::make_unique<Encoder>(weights_ready);

}
float* Trainer::backwardCompatibility(cudaStream_t requested_stream,
                                      float* output_gradient,
                                      int batch,
                                      int hidden,
                                      int sequence){
    if(!trainable){
        throw std::logic_error("Backward is only available for the student trainer.");
    }
    if(train_stream != requested_stream){
        throw std::invalid_argument("Backward requested with a stream not owned by this trainer.");
    }
    return encoder_layer->backwardEnc(output_gradient,d_modelWeights,batch,hidden,sequence);
}
float* Trainer::forward(float* query,
                        float* key,
                        float* value,
                        float* rope_cos,
                        float* rope_sin,
                        int batch,
                        int hidden,
                        int sequence,
                        int head_dim){
    return encoder_layer->forwardEnc(
        query,key,value,rope_cos,rope_sin,batch,hidden,sequence,head_dim);
}
KwD::KwD(){
    cudaStreamCreate(&student_stream); 
    cudaStreamCreate(&teacher_stream);
    cudaStreamCreate(&memOps_stream);

    cudaEventCreate(&device_weights_ready);
}
KwD::~KwD(){
    student.reset();
    teacher.reset();
    if(student_weights != nullptr){
        cudaFreeAsync(student_weights,memOps_stream);
    }
    if(teacher_weights != nullptr){
        cudaFreeAsync(teacher_weights,memOps_stream);
    }
    cudaStreamSynchronize(memOps_stream);
    if(device_weights_ready != nullptr){
        cudaEventDestroy(device_weights_ready);
    }
    if(model_memPool != nullptr){
        cudaMemPoolDestroy(model_memPool);
    }
    cudaStreamDestroy(student_stream);
    cudaStreamDestroy(teacher_stream);
    cudaStreamDestroy(memOps_stream);
}
void KwD::load_model(const std::string& model_path, size_t totalElements){
//freaking confused
    
    std::ifstream model_weights(model_path, std::ios::binary);
    m_weightBytes = totalElements * sizeof(float);
    float* host_pinned_weights = nullptr;
    cudaHostAlloc(&host_pinned_weights, m_weightBytes, cudaHostAllocDefault);
    model_weights.read(reinterpret_cast<char*>(host_pinned_weights), m_weightBytes);   

    cudaMemPoolProps    poolProps={};
                        poolProps.allocType = cudaMemAllocationTypePinned;
                        poolProps.location.type = cudaMemLocationTypeDevice;
                        poolProps.location.id = 0;
    cudaMemPoolCreate(&model_memPool,&poolProps);
    uint64_t threshold=1024*1024*1024;
    cudaMemPoolSetAttribute(model_memPool,cudaMemPoolAttrReleaseThreshold,&threshold);
    cudaMallocAsync(&student_weights,m_weightBytes,model_memPool,memOps_stream);
    cudaMallocAsync(&teacher_weights,m_weightBytes,model_memPool,memOps_stream);

    cudaMemcpyAsync(student_weights,host_pinned_weights,m_weightBytes,cudaMemcpyHostToDevice,memOps_stream);
    cudaMemcpyAsync(teacher_weights,student_weights,m_weightBytes,cudaMemcpyDeviceToDevice,memOps_stream);
    cudaEventRecord(device_weights_ready,memOps_stream);
    cudaStreamSynchronize(memOps_stream);
    cudaFreeHost(host_pinned_weights); 

    student=std::make_unique<Trainer>(student_weights,device_weights_ready,student_stream,true);
    teacher=std::make_unique<Trainer>(teacher_weights,device_weights_ready,teacher_stream,false);
}
float* KwD::teacherForward(float* query,
                           float* key,
                           float* value,
                           float* rope_cos,
                           float* rope_sin,
                           int batch,
                           int hidden,
                           int sequence,
                           int head_dim){
    return teacher->forward(
        query,key,value,rope_cos,rope_sin,batch,hidden,sequence,head_dim);
}
float* KwD::studentForward(float* query,
                           float* key,
                           float* value,
                           float* rope_cos,
                           float* rope_sin,
                           int batch,
                           int hidden,
                           int sequence,
                           int head_dim){
    return student->forward(
        query,key,value,rope_cos,rope_sin,batch,hidden,sequence,head_dim);
}
float* KwD::studentBackward(float* output_gradient,int batch,int hidden,int sequence){
    return student->backwardCompatibility(student_stream,output_gradient,batch,hidden,sequence);
}
/*encoder::constructor & destructor*/
Encoder::Encoder(cudaEvent_t weights_ready){
    attn_layer=std::make_unique<ATTN>();
    
    cudaStreamCreate(&ffn_stream);
    cudaStreamCreate(&attn_stream);
    cudaEventCreateWithFlags(&ffn_backward_done,cudaEventDisableTiming);

    cudaStreamWaitEvent(attn_stream,weights_ready,0);
    cudaStreamWaitEvent(ffn_stream,weights_ready,0);
}
Encoder::~Encoder(){
    cudaStreamSynchronize(attn_stream);
    cudaStreamSynchronize(ffn_stream);
    cudaEventDestroy(ffn_backward_done);
    cudaStreamDestroy(ffn_stream);
    cudaStreamDestroy(attn_stream);
}
// Returns fused RoPE attention output and does not run the FFN.
float* Encoder::forwardEnc(float* query,
                           float* key,
                           float* value,
                           float* rope_cos,
                           float* rope_sin,
                           int batch,
                           int hidden,
                           int sequence,
                           int head_dim){
    float* output=attn_layer->forwardAttn(
        query,key,value,rope_cos,rope_sin,batch,hidden,sequence,head_dim,attn_stream);
    checkCuda(cudaStreamSynchronize(attn_stream), "cudaStreamSynchronize(attention output)");
    return output;
}
float* Encoder::backwardEnc(float* output_gradient,
                            float* model_weights,
                            int batch,
                            int hidden,
                            int sequence){
    if(!ffn_layer){
        throw std::logic_error("Backward is unavailable until the FFN kernel is implemented.");
    }
    float* attention_output_gradient=ffn_layer->backwardFFn(output_gradient,model_weights,batch,hidden,sequence,ffn_stream);
    cudaEventRecord(ffn_backward_done,ffn_stream);
    cudaStreamWaitEvent(attn_stream,ffn_backward_done,0);
    float* input_gradient=attn_layer->backwardAttn(attention_output_gradient,model_weights,batch,hidden,sequence,attn_stream);
    return input_gradient;
}
/*forwarded to encoder attn & ffn*/
ATTN::ATTN(){
    checkCu(cuInit(0), "cuInit");
    checkCu(cuModuleLoad(&attn_module, "transfAot\\attn.ptx"), "cuModuleLoad(attn.ptx)");
    try{
        checkCu(cuModuleGetFunction(&attn_kernel,attn_module,"attn_krnl"),
                "cuModuleGetFunction(attn_krnl)");
    }catch(...){
        cuModuleUnload(attn_module);
        attn_module = nullptr;
        throw;
    }
}
ATTN::~ATTN(){
    if(d_attention_output != nullptr){
        cudaError_t status = cudaFree(d_attention_output);
        if(status != cudaSuccess){
            std::cerr << "cudaFree(attention output): " << cudaGetErrorString(status) << '\n';
        }
    }
    if(attn_module != nullptr){
        CUresult status = cuModuleUnload(attn_module);
        if(status != CUDA_SUCCESS){
            const char* error_text = nullptr;
            cuGetErrorString(status, &error_text);
            std::cerr << "cuModuleUnload(attn.ptx): "
                      << (error_text != nullptr ? error_text : "CUDA Driver API error") << '\n';
        }
    }
}
float* ATTN::forwardAttn(float* query,
                         float* key,
                         float* value,
                         float* rope_cos,
                         float* rope_sin,
                         int batch,
                         int hidden,
                         int sequence,
                         int head_dim,
                         cudaStream_t attn_stream){
    if(query == nullptr || key == nullptr || value == nullptr ||
       rope_cos == nullptr || rope_sin == nullptr){
        throw std::invalid_argument("Q, K, V, and RoPE table device pointers must not be null.");
    }
    if(batch <= 0 || hidden <= 0 || sequence <= 0){
        throw std::invalid_argument("batch, hidden, and sequence must be positive.");
    }
    if(head_dim <= 0 || head_dim % 2 != 0 || hidden % head_dim != 0){
        throw std::invalid_argument("head_dim must be positive, even, and divide hidden.");
    }

    const size_t sequence_size = static_cast<size_t>(sequence);
    const size_t hidden_size = static_cast<size_t>(hidden);
    if(sequence_size > std::numeric_limits<size_t>::max() / hidden_size){
        throw std::overflow_error("Attention input size overflows size_t.");
    }
    const size_t input_elements_per_batch = sequence_size * hidden_size;
    if(static_cast<size_t>(batch) >
       std::numeric_limits<size_t>::max() / input_elements_per_batch){
        throw std::overflow_error("Attention input offset overflows size_t.");
    }
    if(static_cast<size_t>(batch) >
       std::numeric_limits<size_t>::max() / input_elements_per_batch / sizeof(float)){
        throw std::overflow_error("Attention output buffer size overflows size_t.");
    }
    const size_t required_bytes =
        static_cast<size_t>(batch) * input_elements_per_batch * sizeof(float);
    if(required_bytes > output_capacity_bytes){
        if(d_attention_output != nullptr){
            checkCuda(cudaFree(d_attention_output), "cudaFree(attention output)");
            d_attention_output = nullptr;
            output_capacity_bytes = 0;
        }
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_attention_output), required_bytes),
                  "cudaMalloc(attention output)");
        output_capacity_bytes = required_bytes;
    }

    constexpr int block_m = 16;
    constexpr int block_n = 16;
    constexpr int block_k = 32;
    constexpr int block_v = 32;
    constexpr int num_warps = 4;
    constexpr int warp_size = 32;
    // Must match the pinned Triton build's compiled kernel metadata.
    constexpr unsigned int shared_memory_bytes = 36864;
    int m = sequence;
    int n = sequence;
    int k = hidden;
    int stride_am = hidden;
    int stride_bn = hidden;
    int stride_cm = hidden;
    CUdeviceptr unused_global_scratch = 0;
    CUdeviceptr unused_profile_scratch = 0;
    const unsigned int grid_m =
        (static_cast<unsigned int>(sequence) + block_m - 1) / block_m;
    const unsigned int grid_v =
        (static_cast<unsigned int>(hidden) + block_v - 1) / block_v;

    for(int batch_index = 0; batch_index < batch; ++batch_index){
        const size_t input_offset =
            static_cast<size_t>(batch_index) * input_elements_per_batch;
        const size_t output_offset = input_offset;
        CUdeviceptr q_ptr = reinterpret_cast<CUdeviceptr>(query + input_offset);
        CUdeviceptr k_ptr = reinterpret_cast<CUdeviceptr>(key + input_offset);
        CUdeviceptr v_ptr = reinterpret_cast<CUdeviceptr>(value + input_offset);
        CUdeviceptr cos_ptr = reinterpret_cast<CUdeviceptr>(rope_cos);
        CUdeviceptr sin_ptr = reinterpret_cast<CUdeviceptr>(rope_sin);
        CUdeviceptr output_ptr =
            reinterpret_cast<CUdeviceptr>(d_attention_output + output_offset);

        // Triton appends global- and profile-scratch pointers to the PTX entry ABI.
        void* args[] = {
            &m,
            &n,
            &k,
            &head_dim,
            &stride_am,
            &stride_bn,
            &stride_cm,
            &q_ptr,
            &k_ptr,
            &v_ptr,
            &cos_ptr,
            &sin_ptr,
            &output_ptr,
            &unused_global_scratch,
            &unused_profile_scratch,
        };
        checkCu(cuLaunchKernel(attn_kernel,
                               grid_m,
                               grid_v,
                               1,
                               num_warps * warp_size,
                               1,
                               1,
                               shared_memory_bytes,
                               reinterpret_cast<CUstream>(attn_stream),
                               args,
                               nullptr),
                "cuLaunchKernel(attn_krnl)");
    }
    return d_attention_output;
}
float* ATTN::backwardAttn(float* output_gradient,
                          float* model_weights,
                          int batch,
                          int hidden,
                          int sequence,
                          cudaStream_t attn_stream){
    return nullptr;
}
FFN::FFN(){
    cuModuleLoad(&ffn_module, "ffn.ptx");
    cuModuleGetFunction(&ffn_kernel,ffn_module,"ffn_kernel");
}
FFN::~FFN(){
    cuModuleUnload(ffn_module);
}
float* FFN::forwardFFn(float* attention_output,
                       float* model_weights,
                       int batch,
                       int hidden,
                       int sequence,
                       cudaStream_t ffn_stream){
    return nullptr;
}
float* FFN::backwardFFn(float* output_gradient,
                        float* model_weights,
                        int batch,
                        int hidden,
                        int sequence,
                        cudaStream_t ffn_stream){
    return nullptr;
}




}//nameSpace Bracket__here it is

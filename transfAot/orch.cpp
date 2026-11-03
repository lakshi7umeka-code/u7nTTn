#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <vector>
#include "orch.hpp"

namespace transfAot {
Trainer::Trainer(float* model_weights,
                 cudaEvent_t weights_ready,
                 cudaStream_t train_stream,
                 bool trainable)
    : d_modelWeights(model_weights),
      train_stream(train_stream),
      execution_stream(train_stream),
      trainable(trainable){
    cudaStreamWaitEvent(execution_stream,weights_ready,0);
    encoder_layer=std::make_unique<Encoder>();

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
float* Trainer::forward(float* input, int batch, int hidden, int sequence){
    return encoder_layer->forwardEnc(input,d_modelWeights,batch,hidden,sequence);
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
float* KwD::teacherForward(float* input,int batch,int hidden,int sequence){
    return teacher->forward(input,batch,hidden,sequence);
}
float* KwD::studentForward(float* input,int batch,int hidden,int sequence){
    return student->forward(input,batch,hidden,sequence);
}
float* KwD::studentBackward(float* output_gradient,int batch,int hidden,int sequence){
    return student->backwardCompatibility(student_stream,output_gradient,batch,hidden,sequence);
}
/*encoder::constructor & destructor*/
Encoder::Encoder(){
    ffn_layer=std::make_unique<FFN>();
    attn_layer=std::make_unique<ATTN>();
    
    cudaStreamCreate(&ffn_stream);
    cudaStreamCreate(&attn_stream);

}
Encoder::~Encoder(){
    cudaStreamDestroy(ffn_stream);
    cudaStreamDestroy(attn_stream);
}
//link encoder with forward=attn+ffn
float* Encoder::forwardEnc(float* d_in,
                           float* model_weights,
                           int batch,
                           int hidden,
                           int sequence){
    float* attn_out=attn_layer->forwardAttn(d_in,model_weights,batch,hidden,sequence,attn_stream);
    float* ffn_out=ffn_layer->forwardFFn(attn_out,model_weights,batch,hidden,sequence,ffn_stream);
    return ffn_out;
}
float* Encoder::backwardEnc(float* output_gradient,
                            float* model_weights,
                            int batch,
                            int hidden,
                            int sequence){
    float* attention_output_gradient=ffn_layer->backwardFFn(output_gradient,model_weights,batch,hidden,sequence,ffn_stream);
    float* input_gradient=attn_layer->backwardAttn(attention_output_gradient,model_weights,batch,hidden,sequence,attn_stream);
    return input_gradient;
}
/*forwarded to encoder attn & ffn*/
ATTN::ATTN(){
    cuModuleLoad(&attn_module, "attn.ptx");
    cuModuleGetFunction(&attn_kernel,attn_module,"attn_kernel");   
}
ATTN::~ATTN(){
    cuModuleUnload(attn_module);
}
float* ATTN::forwardAttn(float* d_in,
                         float* model_weights,
                         int batch,
                         int hidden,
                         int sequence,
                         cudaStream_t attn_stream){
    return nullptr;
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

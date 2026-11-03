#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <fstream>
#include <vector>
#include "orch.hpp"

namespace transfAot {
Trainer::Trainer(KwD* model_weights,cudaStream_t train_stream){
    d_targetPtr = model_weights->getDevicePointer();
    this->train_stream = train_stream;
    this->execution_stream = train_stream;
    cudaEvent_t model_weights_ready=model_weights->getReadyEvent();
    cudaStreamWaitEvent(execution_stream,model_weights_ready,0);
    encoder_layer=std::make_unique<Encoder>();
}
void Trainer::backwardCompatibility(cudaStream_t student_stream,float* output_gradient,int batch,int hidden,int sequence){
    if(this->train_stream==student_stream){
        backward(output_gradient, batch, hidden, sequence);
    }
}
float* Trainer::forward(float* input, int batch, int hidden, int sequence){
    return encoder_layer->forwardEnc(input, batch, hidden, sequence);
}
void Trainer::backward(float* output_gradient, int batch, int hidden, int sequence){
    encoder_layer->backwardEnc(output_gradient);
}
KwD::KwD(){
    cudaStreamCreate(&student_stream); 
    cudaStreamCreate(&teacher_stream);
    cudaStreamCreate(&memOps_stream);

    cudaEventCreate(&device_weights_ready);
}
KwD::~KwD(){
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
    cudaMallocAsync(&device_weights,m_weightBytes,model_memPool,memOps_stream);
    
    cudaMemcpyAsync(device_weights,host_pinned_weights, m_weightBytes, cudaMemcpyHostToDevice, memOps_stream);
    cudaEventRecord(device_weights_ready,memOps_stream);
    cudaStreamSynchronize(memOps_stream);
    cudaFreeHost(host_pinned_weights); 

    student=std::make_unique<Trainer>(this,student_stream);
    teacher=std::make_unique<Trainer>(this,teacher_stream);
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
float* Encoder::forwardEnc(float* d_in,int b,int h,int s){//b,h,d,s
    float* attn_out=attn_layer->forwardAttn(d_in,attn_stream);
    float* ffn_out=ffn_layer->forwardFFn(attn_out,ffn_stream);
    return ffn_out;
}
float* Encoder::backwardEnc(float* output_gradient){
    float* attention_output_gradient=ffn_layer->backwardFFn(output_gradient,ffn_stream);
    float* input_gradient=attn_layer->backwardAttn(attention_output_gradient,attn_stream);
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
float* ATTN::forwardAttn(float* d_in,cudaStream_t attn_stream){
    return nullptr;
}
float* ATTN::backwardAttn(float* output_gradient,cudaStream_t attn_stream){
    return nullptr;
}
FFN::FFN(){
    cuModuleLoad(&ffn_module, "ffn.ptx");
    cuModuleGetFunction(&ffn_kernel,ffn_module,"ffn_kernel");
}
FFN::~FFN(){
    cuModuleUnload(ffn_module);
}
float* FFN::forwardFFn(float* FW_attn_out,cudaStream_t ffn_stream){
    return nullptr;
}
float* FFN::backwardFFn(float* output_gradient,cudaStream_t ffn_stream){
    return nullptr;
}




}//nameSpace Bracket__here it is

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
}
void Trainer::backwardCompatibility(cudaStream_t student_stream){
    if(this->train_stream==student_stream){
        backward(this->train_stream);
    }
}
KwD::KwD(){
    encoder_layer=std::make_unique<Encoder>();

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
    float* attn_out=attn_layer->forwardAttn(/*d_in,attn_stream*/);
    float* ffn_out=ffn_layer->forwardFFn(/*attn_out,ffn_stream*/);
}
float* Encoder::backwardEnc(){

}
/*forwarded to encoder attn & ffn*/
ATTN::ATTN(){
    cuModuleLoad(&attn_module, "attn.ptx");
    cuModuleGetFunction(&attn_kernel,attn_module,"attn_kernel");   
}
ATTN::~ATTN(){
    cuModuleUnload(attn_module);
}
float* ATTN::forwardAttn(){}
float* ATTN::backwardAttn(){}
FFN::FFN(){
    cuModuleLoad(&ffn_module, "ffn.ptx");
    cuModuleGetFunction(&ffn_kernel,ffn_module,"ffn_kernel");
}
FFN::~FFN(){
    cuModuleUnload(ffn_module);
}
float* FFN::forwardFFn(){}
float* FFN::backwardFFn(){}

}//nameSpace Bracket__here it is

#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include "orch.hpp"

namespace transfAot {

KwD::KwD(){
    student_stream=std::make_unique<Trainer>();
    teacher_stream=std::make_unique<Trainer>();
    encoder_layer=std::make_unique<Encoder>();
}
Encoder::Encoder(){
    ffn_layer=std::make_unique<FFN>();
    attn_layer=std::make_unique<ATTN>();
}

}
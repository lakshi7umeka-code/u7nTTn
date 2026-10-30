#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#include <cuda.h>

namespace transfAot{
class Trainer{
    private:
};
class Encoder{
    private:
        std::unique_ptr<FFN> ffn_layer;
        std::unique_ptr<ATTN> attn_layer;
};
class KwD{
    private:
        std::unique_ptr<Trainer> student_stream, teacher_stream;
        std::unique_ptr<Encoder> encoder_layer;
    public:
        KwD();
    };
}


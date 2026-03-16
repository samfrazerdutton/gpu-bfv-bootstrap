# Kernels

CUDA kernels for RNS arithmetic and EvalMod.

## Compile

    SM=sm_75  # change to match your GPU
    nvcc --ptx -arch=$SM -O3 rns_base_conv.cu -o rns_base_conv_${SM}.ptx
    nvcc --ptx -arch=$SM -O3 eval_mod.cu      -o eval_mod_${SM}.ptx

Pre-compiled PTX files are excluded from git (*.ptx in .gitignore).

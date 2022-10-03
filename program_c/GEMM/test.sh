nvcc GEMM.cu -o results/GEMM -lcublas
echo "compiler over!"
echo "profiler test GEMM"
nvprof ./results/GEMM 12 12 12 7
# -arch=native      : build for whatever GPU this Colab session has
# --extended-lambda : allow __host__ __device__ lambdas inside functions
# -o main           : name the compiled program "main"
# && ./main         : if compiling succeeded, run the program
!nvcc -arch=native --extended-lambda -o main main.cu && ./main
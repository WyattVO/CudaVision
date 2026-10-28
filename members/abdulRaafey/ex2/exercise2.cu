thrust::for_each(thrust::device, vec.begin(), vec.end(), [] __device__ (int val)){
    printf("printing %d on %s\n", val, cv::execution_space());
}
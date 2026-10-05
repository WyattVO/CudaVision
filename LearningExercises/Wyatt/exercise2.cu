thrust::for_each(thrust::host, vec.begin(), vec.end(), [] __host__ (int val) {
  printf("printing %d on %s\n", val, cv::execution_space());
});
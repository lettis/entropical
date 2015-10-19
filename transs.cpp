
#ifdef TRANSS_OCL
  #include "transs_ocl.hpp"
#else
  #include "transs_omp.hpp"
#endif

int main(int argc, char* argv[]) {
#ifdef TRANSS_OCL
  return main_omp(int argc, char* argv[]);
#else
  return main_ocl(int argc, char* argv[]);
#endif
}


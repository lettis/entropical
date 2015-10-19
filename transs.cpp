
#ifdef TRANSS_OCL
  #include "transs_ocl.hpp"
#else
  #include "transs_omp.hpp"
#endif

int main(int argc, char* argv[]) {
#ifdef TRANSS_OCL
  return main_ocl(argc, argv);
#else
  return main_omp(argc, argv);
#endif
}


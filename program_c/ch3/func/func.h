int func(void (*p)(int *, int *, unsigned int), int *g_idata, int *g_odata,
        int *h_idata, int *h_odata, unsigned int n, size_t nBytes, dim3 block, dim3 grid, const char *s);

int func_nested(void (*p)(int *, int *, unsigned int, int, int), int *g_idata, int *g_odata, int stride,
        int *h_idata, int *h_odata, unsigned int n, size_t nBytes, dim3 block, dim3 grid, const char *s);
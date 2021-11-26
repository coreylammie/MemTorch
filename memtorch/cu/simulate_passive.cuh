__device__ double cumsum(int *p, int *c, int n) {
  int nz = 0;
  double nz2 = 0;
  for (int i = 0; i < n; i++) {
    p[i] = nz;
    nz += c[i];
    nz2 += c[i];
    c[i] = p[i];
  }
  p[n] = nz;
  return nz2;
}
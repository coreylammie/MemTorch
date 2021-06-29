template <class T> T min_(T a, T b) { return !(b < a) ? a : b; };
template <class T> T max_(T a, T b) { return (a<b)?b:a; };
template <class T> T clamp_(T x, T min, T max) {
  if (x < min)
    x = min;
  if (x > max)
    x = max;
  return x;
}
void merge_sort(float *arr, int low, int high);

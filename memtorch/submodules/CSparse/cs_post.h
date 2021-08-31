/* post order a forest */
CUDA_CALLABLE_MEMBER
csi *cs_post(const csi *parent, csi n) {
  csi j, k = 0, *post, *w, *head, *next, *stack;
  if (!parent)
    return (NULL);                                /* check inputs */
  post = (ptrdiff_t *)malloc(sizeof(csi) * n);  /* allocate result */
  w = (ptrdiff_t *)malloc(sizeof(csi) * (3 * n)); /* get workspace */
  if (!w || !post)
    return (cs_idone(post, NULL, w, 0));
  head = w;
  next = w + n;
  stack = w + 2 * n;
  for (j = 0; j < n; j++)
    head[j] = -1;              /* empty linked lists */
  for (j = n - 1; j >= 0; j--) /* traverse nodes in reverse order*/
  {
    if (parent[j] == -1)
      continue;                /* j is a root */
    next[j] = head[parent[j]]; /* add j to list of its parent */
    head[parent[j]] = j;
  }
  for (j = 0; j < n; j++) {
    if (parent[j] != -1)
      continue; /* skip j if it is not a root */
    k = cs_tdfs(j, k, head, next, post, stack);
  }
  return (cs_idone(post, NULL, w, 1)); /* success; free w, return post */
}

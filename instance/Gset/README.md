# Stanford Gset

This directory contains the 71 graph instances published in the Stanford Gset
collection:

https://web.stanford.edu/~yyye/yyye/Gset/

The source files are stored as `G<number>.txt` so they can be selected directly
with `--graph Gset --Gset_id <number>`. Each file begins with the number of
vertices and edges, followed by one weighted edge per line:

```text
<vertex_count> <edge_count>
<u> <v> <weight>
```

Included instance IDs: 1-67, 70, 72, 77, and 81.

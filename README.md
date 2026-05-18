# BinaryLLM

Experimental binary-native LLM architecture for edge devices.
C++ core compiled to native and WebAssembly. No floating-point in inference.
Local learning rules instead of backpropagation.

## Phase 1 status

- [x] Project skeleton
- [ ] `BinaryVec<N>` template with specializations
- [ ] Token encoder (word + symbol namespaces)
- [ ] Tokenizer (string → token stream)
- [ ] Corpus collision validation

## Build

```bash
cmake --preset debug
cmake --build build/debug
ctest --preset debug
```

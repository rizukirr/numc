# Contributing to numc

First off, thank you for considering contributing to `numc`! It's people like you that make `numc` such a high-performance and robust library.

## Code of Conduct

By participating in this project, you agree to abide by our professional standards:
- Focus on technical excellence and performance.
- Be respectful and constructive in reviews and discussions.
- Prioritize library stability and correctness.

## How Can I Contribute?

### Reporting Bugs
Before creating a bug report, please check the existing issues to see if the problem has already been reported. When reporting a bug, include:
- A clear, descriptive title.
- Steps to reproduce the issue.
- A minimal code example (C or Python benchmark).
- Details about your environment (OS, CPU, Compiler version).

### Suggesting Enhancements
We welcome ideas for performance optimizations and new tensor operations. Please open an issue to discuss major changes before starting implementation.

### Pull Requests
1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests in `tests/`.
3. If you've changed APIs, update the documentation in `include/numc/` and the Wiki.
4. Ensure the test suite passes.
5. Make sure your code follows the project's style (see below).

## Development Workflow

We provide a `run.sh` script to simplify common tasks:

```bash
./run.sh release    # Build optimized version
./run.sh debug      # Build with debug symbols and ASan
./run.sh test       # Run all tests
./run.sh bench      # Run performance benchmarks
```

## Style Guidelines

### Code Formatting
This project strictly enforces `clang-format`. Please run the following before committing:

```bash
find src include tests examples bench -iname *.c -o -iname *.h | xargs clang-format -i
```

### Technical Standards
- **Standard:** Pure **C23**. Use modern features where they improve clarity or safety.
- **Safety:** Always check for `NULL` pointers and valid shapes. Use `NUMC_SET_ERROR`.
- **Memory:** Prefer the arena allocator (`NumcCtx`) for all tensor data. Avoid raw `malloc`/`free` inside core kernels.
- **Performance:** 
    - Use multi-accumulation in loops to avoid dependency chains.
    - Leverage OpenMP (`NUMC_OMP_FOR`) for data-parallel operations.
    - Offload to BLIS for Level 3 BLAS operations where possible.

## CI Pipeline
Every Pull Request triggers our CI pipeline, which checks:
1. **Formatting:** Matches `.clang-format`.
2. **Static Analysis:** Passes `clang-tidy` checks.
3. **Build Matrix:** Compiles on GCC and Clang with/without BLAS.
4. **Tests:** All tests must pass, including AddressSanitizer checks.

Thank you for your contributions!

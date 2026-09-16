# Fusion baselines

Per-device baselines for `test-fusion`, one CSV per backend (e.g. `MTL.csv`). Rows are
`arch,moe,mode,label,count`. Regenerate a CSV whenever fusion patterns change.

## Update a baseline

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON   # enable the target backend
cmake --build build --config Release --target test-llama-archs --target test-fusion -j

rm -rf build-ci-models && mkdir -p build-ci-models
./build/bin/test-llama-archs -o build-ci-models

./build/bin/test-fusion --models build-ci-models --device MTL0 --record MTL.csv
```

## Validate

```sh
./build/bin/test-fusion --models build-ci-models --device MTL0 --check MTL.csv
```

Checks compare row presence as well as counts, so a disappeared fusion or missing
model in a directory run fails. Use `--model FILE` to check only that architecture's
rows against a larger baseline. Nonfinite results and failed model/reference runs
also fail, even if no fusion counters were emitted. An unavailable device returns
77 (not a pass). `test-fusion-check` exercises the row-set and numeric predicates
without requiring a model or GPU.

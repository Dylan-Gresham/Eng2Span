#!/bin/bash

if [ -f "data/benchmarks.csv" ]; then
    uv run python -c "import pandas as pd; df = pd.read_csv('benchmarks.csv'); df = df.set_index('Unnamed: 0'); df.index.names = ['Model']; print(df.to_markdown(index=True))"
else
    echo "Benchmarks haven't been generated yet."
    echo "Run the \`gen-metrics.sh\` script before this one."
fi

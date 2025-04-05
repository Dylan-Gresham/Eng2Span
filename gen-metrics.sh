#!/bin/bash

if [ "$(hostname)" == "ENG402758" ]; then
    uv run src/m2m_baseline.py &
    uv run src/m2m_ft.py &

    uv run src/mbart_baseline.py &
    uv run src/mbart_ft.py &

    uv run src/nllb_baseline.py &
    uv run src/nllb_ft.py &

    uv run src/opus_baseline.py &
    uv run src/opus_ft.py &

    wait

    uv run src/join_eval_metrics.py

    uv run python -c "
    import pandas as pd
    df = pd.read_csv('data/benchmarks.csv')
    print(df.to_markdown(index=True))
    "
else
    echo "This script is only setup to run on Boise State's ENG402758 machine."
    echo "* The script assumes the device is running with 4 20gb VRAM GPUs and won't work on other configurations without modification."
    
    exit(1)
fi

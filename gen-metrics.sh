#!/bin/bash

if [ "$(hostname)" == "ENG402758" ]; then
    echo "Generating baseline scores..."

    uv run src/m2m_baseline.py 0 &
    uv run src/mbart_baseline.py 1 &
    uv run src/nllb_baseline.py 2 &
    uv run src/opus_baseline.py 3 &

    wait

    echo "Baseline scores generated!"

    echo "Generating fine-tuned scores..."

    uv run src/m2m_ft.py 0 &
    uv run src/mbart_ft.py 1 &
    uv run src/nllb_ft.py 2 &
    uv run src/opus_ft.py 3 &

    wait

    echo "Fine-tuned scores generated!"

    uv run src/join_eval_metrics.py

    echo "Results:"

    uv run python -c "
    import pandas as pd
    df = pd.read_csv('data/benchmarks.csv')
    print(df.to_markdown(index=True))
    "
else
    echo "This script is only setup to run on Boise State's ENG402758 machine."
    echo "* The script assumes the device is running with 4 20gb VRAM GPUs and won't work on other configurations without modification."

    exit 1
fi

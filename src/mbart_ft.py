import os
import sys

import evaluate
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    MBartTokenizer,
)

device = "cuda:1" if torch.cuda.is_available() else "cpu"

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
ter = evaluate.load("ter")

data = pd.read_csv("./data/combined.data")
test_split = data.loc[data["split"] == "test"]
test_en = test_split["en"].to_list()
test_en = [str(e) for e in test_en]
references = test_split["es"].to_list()
references = [str(e) for e in references]


results = pd.DataFrame(
    columns=["BLEU", "METEOR", "ROUGE", "TER"],
    index=[],
)

tokenizer = MBartTokenizer.from_pretrained(
    "facebook/mbart-large-50", src_lang="en_XX", tgt_lang="es_XX"
)


def translate_batched(model, en_sentences, position, batch_size=8):
    translations = []
    os.system("clear")
    for i in tqdm(
        range(0, len(en_sentences), batch_size),
        desc="Generating MBart FT translations",
        position=position,
    ):
        encoded = tokenizer(
            en_sentences[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        generated_tokens = model.generate(
            **encoded, forced_bos_token_id=tokenizer.lang_code_to_id["es_XX"]
        ).to("cpu")
        translations.extend(
            tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        )

    return translations


def evaluate(position):
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "models/mbart", local_files_only=True
    ).to(device)
    translations = translate_batched(model, test_en, position)

    bleu_result = bleu.compute(predictions=translations, references=references)["bleu"]
    meteor_result = meteor.compute(predictions=translations, references=references)[
        "meteor"
    ]
    rouge_result = rouge.compute(predictions=translations, references=references)[
        "rougeL"
    ]
    ter_result = ter.compute(predictions=translations, references=references)["score"]
    results.loc["MBart FT"] = {
        "BLEU": bleu_result,
        "METEOR": meteor_result,
        "ROUGE": rouge_result,
        "TER": ter_result,
    }


def run_benchmarks(position):
    evaluate(position)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: python {str(sys.argv[0])} <position>")
        exit(1)

    run_benchmarks(int(sys.argv[1]))

    results.to_csv("data/mbart_ft.csv")

import evaluate
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSeq2SeqLM,
    NllbTokenizer,
)

device = "cuda:2" if torch.cuda.is_available() else "cpu"

bleu = evaluate.load("bleu")
meteor = evaluate.load("meteor")
rouge = evaluate.load("rouge")
ter = evaluate.load("ter")

print("Reading in data")
data = pd.read_csv("./data/combined.data")
test_split = data.loc[data["split"] == "test"]
test_en = test_split["en"].to_list()
test_en = [str(e) for e in test_en]
references = test_split["es"].to_list()
references = [str(e) for e in references]


print("Initializing baseline DataFrame for results")
results = pd.DataFrame(
    columns=["BLEU", "METEOR", "ROUGE", "TER"],
    index=[],
)

tokenizer = NllbTokenizer.from_pretrained(
    "facebook/nllb-200-distilled-600M", src_lang="eng_Latn", tgt_lang="spa_Latn"
)


def translate_batched(model, en_sentences, batch_size=8):
    translations = []
    for i in tqdm(
        range(0, len(en_sentences), batch_size), desc="Generating MBart translations"
    ):
        encoded = tokenizer(
            en_sentences[i : i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        generated_tokens = model.generate(
            **encoded, forced_bos_token_id=tokenizer.get_lang_id("es")
        ).to("cpu")
        translations.extend(
            tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        )

    return translations


def evaluate():
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M"
    ).to(device)
    translations = translate_batched(model, test_en)

    bleu_result = bleu.compute(predictions=translations, references=references)["bleu"]
    meteor_result = meteor.compute(predictions=translations, references=references)[
        "meteor"
    ]
    rouge_result = rouge.compute(predictions=translations, references=references)[
        "rougeL"
    ]
    ter_result = ter.compute(predictions=translations, references=references)["score"]
    results.loc["NLLB FT"] = {
        "BLEU": bleu_result,
        "METEOR": meteor_result,
        "ROUGE": rouge_result,
        "TER": ter_result,
    }


def run_benchmarks():
    evaluate()


if __name__ == "__main__":
    run_benchmarks()
    results.to_csv("data/nllb_baseline.csv", index=False)

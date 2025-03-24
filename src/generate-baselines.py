import evaluate
import pandas as pd
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    M2M100ForConditionalGeneration,
    M2M100Tokenizer,
)

BENCHMARKS = ["bleu", "meteor", "rouge", "ter"]
MODELS = [
    "google/mT5-small",
    "facebook/mbart-large-50",
    "Helsinki-NLP/opus-mt-en-es",
    "facebook/m2m100_418M",
    "facebook/nllb-200-distilled-600M",
]

data = pd.read_csv("./data/combined.data")
test_split = data.loc[data["split"] == "split"]
test_en = test_split["en"].to_list()
references = test_split["es"].to_list()


def get_name_from_repo(repo_str):
    """Gets the name of the model from the Hugging Face repository string."""
    return repo_str.split("/")[1]


baselines = pd.DataFrame(
    columns=["BLEU", "METEOR", "ROUGE", "TER"],
    index=[get_name_from_repo(model) for model in MODELS],
)


def generate_prediction(model, tokenizer, input):
    """Uses the input model and tokenizer to generate a translation of the input."""
    model_input = tokenizer(input, return_tensors="pt")
    return tokenizer.batch_decode(model.generate(model_input), skip_special_tokens=True)


def m2m_generate_prediction(model, tokenizer, input):
    """Uses the input M2M model and tokenizer to generate a translation of the input."""
    tokenizer.src_lang = "en"
    encoded_en = tokenizer(test_en, return_tensors="pt")
    generated_toks = model.generate(
        **encoded_en, forced_bos_token_id=tokenizer.get_land_id("es")
    )

    return tokenizer.batch_decode(
        model.generate(generated_toks), skip_special_tokens=True
    )


def run_bleu_benchmark(model, tokenizer, generator):
    bleu = evaluate.load(BENCHMARKS[0])
    predictions = [generator(model, tokenizer, input) for input in test_en]

    results = bleu.compute(predictions=predictions, references=references)
    model_name = get_name_from_repo(model.config._name_or_path)
    baselines.loc[model_name, "BLEU"] = results["bleu"]


def run_meteor_benchmark(model, tokenizer, generator):
    meteor = evaluate.load(BENCHMARKS[1])
    predictions = [generator(model, tokenizer, input) for input in test_en]

    results = meteor.compute(predictions=predictions, references=references)
    model_name = get_name_from_repo(model.config._name_or_path)
    baselines.loc[model_name, "METEOR"] = results["meteor"]


def run_rouge_benchmark(model, tokenizer, generator):
    rouge = evaluate.load(BENCHMARKS[2])
    predictions = [generator(model, tokenizer, input) for input in test_en]

    results = rouge.compute(predictions=predictions, references=references)
    model_name = get_name_from_repo(model.config._name_or_path)
    baselines.loc[model_name, "ROUGE"] = results["rouge"]


def run_ter_benchmark(model, tokenizer, generator):
    ter = evaluate.load(BENCHMARKS[3])
    predictions = [generator(model, tokenizer, input) for input in test_en]

    results = ter.compute(predictions=predictions, references=references)
    model_name = get_name_from_repo(model.config._name_or_path)
    baselines.loc[model_name, "TER"] = results["ter"]


def run_benchmarks_for(model_repo):
    if model_repo != MODELS[3]:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_repo)
        tokenizer = AutoTokenizer.from_pretrained(model_repo)
        generator = generate_prediction
    else:
        model = M2M100ForConditionalGeneration.from_pretrained(model_repo)
        tokenizer = M2M100Tokenizer.from_pretrained(model_repo)
        generator = m2m_generate_prediction

    run_bleu_benchmark(model, tokenizer, generator)
    run_meteor_benchmark(model, tokenizer, generator)
    run_rouge_benchmark(model, tokenizer, generator)
    run_ter_benchmark(model, tokenizer, generator)


def run_benchmarks():
    for model in MODELS:
        run_benchmarks_for(model)


if __name__ == "__main__":
    run_benchmarks()
    print(baselines)

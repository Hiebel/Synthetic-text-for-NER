from GPT_class import *
from utils import *
import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, BloomTokenizerFast
from carbontracker.tracker import CarbonTracker
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to model for inference")
parser.add_argument("--base_model", type=str, required=True, help="Path to pre-trained model")
parser.add_argument("--top_p", type=float, required=True, help="Top_p value")
parser.add_argument("--temp", type=float, required=True, help="Temperature value")
parser.add_argument("--rep_p", type=float, required=True, help="Repetition penalty value")
parser.add_argument("--index", type=str, required=True, help="Index for generating multiple times with same configuration")
args = parser.parse_args()


model_name = args.model.split("/")[-1]

path_to_pretrained = args.base_model

tokenizer = AutoTokenizer.from_pretrained(path_to_pretrained)

special_tokens = load_special_tokens(bos_and_eos=True) #, path_special_tokens="liste_special_tokens.json") # retirer chemin du fichier si pas besoin d'ajouter des tokens spéciaux supplémentaires
special_tokens_dict = {"additional_special_tokens": special_tokens}
tokenizer.add_special_tokens(special_tokens_dict)

model = GPT.load_from_checkpoint(args.model, model_name=path_to_pretrained, len_tokenizer=len(tokenizer)).to("cuda")

input = tokenizer("<|startoftext|>", return_tensors="pt")
input_ids = input.input_ids.to("cuda")
attention_mask = input.attention_mask.to("cuda")

args_generation = {
    "input_ids": input_ids,
    "do_sample": True,
    "num_beams": 5,
    "top_k": 50,
    "top_p": args.top_p, # [.90, .95, 1.0]
    "max_length": 950,
    "repetition_penalty": args.rep_p, # [1.0, 3.0, 10.0]
    "temperature": args.temp, # [0.8, 1.O, ] --> bug avec 0.8
    "attention_mask": attention_mask,
}

# output_dir = f"outputs_gen/outputs_bloom_1b1_balises/topp={args_generation['top_p']}_rp={args_generation['repetition_penalty']}_temp={args_generation['temperature']}"
output_dir = f"outputs_larges/gpt-fr/llf_{args.index}"
print(output_dir)

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
else:
    1/0

model.eval()
outputs = []
nb_generations = 100

target_token_number = 100000 # (60 000 sans balises / 92 000 avec balises)

tracker = CarbonTracker(
        epochs=1,
        monitor_epochs=1,
        log_dir=f"carbontracker/generation_globales/{model_name}_{target_token_number}-tokens"
)

tokens_out = 0
nb_gens = 0


tracker.epoch_start()

while tokens_out < target_token_number:
#for i in range(nb_generations): 
    output = model.generate(**args_generation)
    for sample in output:
        outputs.append(sample)
        tokens_out += len(sample)
        nb_gens += 1
    #print(tokens_out)

tracker.epoch_end()

print(f"Nombre de générations {nb_gens}")

for i, sample in enumerate(outputs):
    texte = tokenizer.decode(sample, skip_special_tokens=False)
    with open(f"{output_dir}/{i}.txt", "w", encoding="utf-8") as fout:
            fout.write(texte)


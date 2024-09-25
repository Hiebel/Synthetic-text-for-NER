from utils import *
from GPT_class import *
from data_class import DataProcess

path_to_model = "../pre_trained_models/bigscience/bloom-7b1"
#path_to_model = "../pre_trained_models/LLF/gpt-base"

#data = lire_e3c_annotations()  # _annotations() pour version xml

data = get_train_e3c() + get_train_cas()

print(len(data))

Data = DataProcess(path_to_model, block_size=128)

special_tokens = load_special_tokens(bos_and_eos=True) #, path_special_tokens="liste_special_tokens.json") # retirer chemin du fichier si pas besoin d'ajouter des tokens spéciaux supplémentaires
special_tokens_dict = {"additional_special_tokens": special_tokens}

train_loader, val_loader, len_tokenizer = Data.prepare_data(data, special_tokens=special_tokens_dict, batch_size=4)

#print(len_tokenizer)

model = GPT(path_to_model, len_tokenizer)

max_epochs = 10

model.set_carbon_tracking(max_epochs=max_epochs, log_dir="carbontracker_prompts/train_bloom_1b1", monitor_epochs=-1) 

model.start_training(train_loader=train_loader, val_loader=val_loader, epochs=max_epochs)

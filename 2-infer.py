# python 2-infer.py

# JIJ: Hoe meldt iemand zich aan bij de tuinvereniging?
# LLM: Door een aanvraag te doen en te voldoen aan de voorwaarden.
# NOT: Om lid te worden van de tuinclub, moet je een aanvraagformulier invullen dat verkrijgbaar is bij het kantoor van de tuinclub of op hun website. Het formulier vraagt doorgaans om basisinformatie zoals je naam, adres, telefoonnummer en e-mailadres. Je moet ook betalen voor het lidmaatschap, dat meestal jaarlijks is. Zodra je aanvraag is ontvangen en goedgekeurd, word je verwelkomd bij de tuinclub en ontvang je een bevestiging van je lidmaatschap.

# JIJ: Wat is de definitie van een 'volkstuin' volgens de statuten?
# LLM: Een volkstuin is een aan een lid op een tuinpark ter beschikking gestelde tuin.
# NOT: Volgens de statuten is een volkstuin een 'door de gemeenschap beheerde tuin waar individuele leden hun eigen groenten, bloemen en fruit kunnen laten groeien en verzorgen'.

# JIJ: Hoeveel gasflessen zijn maximaal toegestaan op een volkstuinpark met vergunning om te overnachten?
# LLM: Op volkstuinparken met een vergunning om te mogen overnachten zijn maximaal 2 gasflessen (waterinhoud 26 liter per fles) toegestaan. Voor volkstuinparken in Amsterdam en Ouder-Amstel is ontheffing verleend tot een maximum van 4 gasflessen.
# NOT: De maximale hoeveelheid propaangrills die zijn toegestaan op een camping met een overnachtingsvergunning, is afhankelijk van de specifieke voorschriften van de camping. Over het algemeen geldt dat hoe groter de camping is, hoe meer propaangrills er zijn toegestaan. De meeste campings met een overnachtingsvergunning staan maximaal twee propaangrills toe, maar sommige kunnen er meer toestaan. Het is het beste om de specifieke voorschriften van de camping te controleren om zeker te zijn van het aantal toegestane propaangrills.

# JIJ: Voor volkstuinparken in Amsterdam en Ouder-Amstel is ontheffing
# LLM: Voor volkstuinparken in Amsterdam en Ouder-Amstel is ontheffing verleend tot een maximum van 4 gasflessen.
# NOT: Voor volkstuinparken in Amsterdam en Ouder-Amstel is ontheffing vereist voor het houden van bijen.
# NOT: Voor volkstuinparken in Amsterdam en Ouder-Amstel is ontheffing vereist van de verbodsbepalingen in de Algemene Plaatselijke Verordening (APV) met betrekking tot het gebruik van grond en gebouwen voor recreatieve doeleinden. De ontheffing wordt verleend door het college van burgemeester en wethouders van Amsterdam of Ouder-Amstel.

# JIJ: Hoeveel gasflessen mogen?
# LLM: Twee. Of vier.
# NOT: Het is wettelijk toegestaan om maximaal 10 gasflessen in een voertuig te vervoeren. Gasflessen moeten worden vervoerd in een goedgekeurde gasfleshouder.

# JIJ: Wat betekent 'algemeen werk' bij de Bond van Volkstuinders?
# LLM: Algemeen werk verwijst naar de taken die door de afdelingsvergaderingen worden vastgesteld en door de leden van een afdeling voor het tuinpark moeten worden uitgevoerd.
# NOT: Algemeen werk is het werk dat door de leden van de vereniging wordt verricht ten behoeve van de vereniging als geheel.

# JIJ: Wat zijn ongeldige stemmen volgens de statuten van de Bond van Volkstuinders?
# LLM: Ongeldige stemmen zijn stemmen uitgebracht door een geschorste stemgerechtigde, en bij schriftelijke stemming ook blanco stemmen en stemmen die een andere aanduiding bevatten dan voor de desbetreffende stemming noodzakelijk is.
# NOT: Ongeldige stemmen zijn stemmen die niet in aanmerking worden genomen bij het tellen van de stemmen voor een verkiezing. In de statuten van de National Gardeners' Federation zijn ongeldige stemmen stemmen die niet zijn uitgebracht op een kandidaat van de National Gardeners' Federation of op een onderschreven kandidaat.


# JIJ: Wat zijn de doelstellingen van VTV Water-Land?
# LLM: De doelstellingen zijn het bevorderen van duurzaam tuinieren en het versterken van de gemeenschapszin.
# NOT: De doelstellingen van VTV Water-Land zijn om de waterkwaliteit en het watergebruik in de regio te verbeteren. VTV Water-Land is een partnerschap tussen de Virginia Tech University en de gemeenschappen van Southwest Virginia. Het programma heeft als doel om de waterkwaliteit te verbeteren door middel van onderzoek, onderwijs en uitvoering.

import time
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from enum import Enum

class LoadingOption(Enum):
    OPTION_1_LoadBaseModelOnly = 1
    OPTION_2_LoadBaseModelAndAdapter = 2
    OPTION_3_LoadMergedModel = 3


def load_model(loading_option):
    # Load the base model without fine-tuned adapter.
    if loading_option == LoadingOption.OPTION_1_LoadBaseModelOnly:
        model = AutoModelForCausalLM.from_pretrained('Rijgersberg/GEITje-7B-chat-v2', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained('Rijgersberg/GEITje-7B-chat-v2')

    # Load the base model with fine-tuned adapter.
    elif loading_option == LoadingOption.OPTION_2_LoadBaseModelAndAdapter:
        base_model = AutoModelForCausalLM.from_pretrained('Rijgersberg/GEITje-7B-chat-v2', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained('./model/adapter')
        model = PeftModel.from_pretrained(base_model, './model/adapter')

    # Load the merged model.
    elif loading_option == LoadingOption.OPTION_3_LoadMergedModel:
        model = AutoModelForCausalLM.from_pretrained('./model/full', torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained('./model/full')

    tokenizer.padding_side = 'right'
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.pad_token_id = tokenizer.unk_token_id
    model.config.pad_token_id = tokenizer.unk_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_eos_token = True
    return model, tokenizer

def generate(conversation, temperature=0.2, top_k=50, max_new_tokens=265):
    tokenized = tokenizer.apply_chat_template(conversation, add_generation_prompt=True, return_tensors='pt')
    outputs = model.generate(tokenized, do_sample=True, temperature=temperature, top_k=top_k, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def chat(user_input):
    conversation = [
        {
            'role': 'user',
            'content': user_input
        }
    ]
    return generate(conversation)


# -----------------------------------------------------------------------------------------------
#
# Choose an option to load the base model, adapater or merged model.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
start_time = time.time()
loading_option = LoadingOption.OPTION_2_LoadBaseModelAndAdapter # <------------------------------
model, tokenizer = load_model(loading_option)
print(f"Loading model time taken: {round(time.time() - start_time)}s")

# print("\n\nExample generated response:\n")
# print(f"LLM: {chat('Wat zijn de doelstellingen van VTV Water-Land?')}\n\n")

while True:
    start_time = time.time()
    user_input = input("JIJ: ")
    if user_input.lower() in ["exit", "stop", "quit", ":q"]:
        print("LLM: Tot ziens!")
        break
    response = chat(user_input)
    print(f"LLM: {response} ({round(time.time() - start_time)}s)\n\n")

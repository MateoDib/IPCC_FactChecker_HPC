from transformers import AutoTokenizer, pipeline, BitsAndBytesConfig
import torch

# Configuration du modèle
model_id = "meta-llama/Meta-Llama-3.1-70B-Instruct"

# Configuration pour BitsAndBytes avec calcul en float16
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,  # Réduction du type de données pour accélérer le calcul
    llm_int8_enable_fp32_cpu_offload=False  # Déchargement des poids sur le CPU si nécessaire
)

# Charger le tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token_id = tokenizer.eos_token_id  # Ajouter le pad_token_id explicitement
tokenizer.padding_side = "left"  # Correction pour decoder-only models comme Llama

# Chargement du pipeline
pipe = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={
        "device_map": "auto",
        "quantization_config": quantization_config,
        "offload_folder": "/tmp",
        "pad_token_id": tokenizer.pad_token_id  # Correction du warning
    },
    tokenizer=tokenizer  # Associer explicitement le tokenizer
)

# Vérification de la répartition des couches
print("Répartition des couches sur les GPUs :")
if hasattr(pipe.model, "hf_device_map"):
    print(pipe.model.hf_device_map)
else:
    print("Aucune répartition détectée.")

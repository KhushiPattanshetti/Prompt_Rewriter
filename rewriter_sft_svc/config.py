"""
config.py — rewriter_sft_svc
"""

import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SVC_DIR = os.path.dirname(__file__)

DATASET_PATH = os.path.join(ROOT_DIR, "data", "structured_notes.csv")
CHECKPOINT_DIR = os.path.join(ROOT_DIR, "sft_checkpoints", "phi3_rewriter_sft")
TEST_DATA_DIR = os.path.join(SVC_DIR, "test_data")
TESTING_SUMMARY_PATH = os.path.join(SVC_DIR, "testing_summary.md")

BASE_MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
TRUST_REMOTE_CODE = True

# Hugging Face adapter repo
# Example:
# export HF_ADAPTER_REPO="ishanmane/phi3-rewriter-sft"
# export HF_TOKEN="hf_xxxxxxxxxxxxxxxxx"
HF_ADAPTER_REPO = os.environ.get("HF_ADAPTER_REPO", "ishanmane/phi3-rewriter-sft")
HF_TOKEN = os.environ.get("HF_TOKEN", None)

SYSTEM_INSTRUCTION = """You are a clinical note rewriter.
Convert the given clinical note into a strict structured clinical note format.

Rules:
1. Extract only explicitly mentioned information.
2. Do not infer, assume, or hallucinate.
3. Do not add explanations, commentary, or reasoning.
4. Do not paraphrase or summarize the note.
5. Follow the required structured format exactly.
6. If a field is missing, write: Not specified.

Required output format:

Patient Demographics:
Age:
Gender:

Primary Diagnosis:
Secondary Diagnoses:
Symptoms:
Duration:
Investigations:
Procedures:
Comorbidities:
Risk Factors:
Complications:
"""

REQUIRED_FIELDS = [
    "Patient Demographics:",
    "Age:",
    "Gender:",
    "Primary Diagnosis:",
    "Secondary Diagnoses:",
    "Symptoms:",
    "Duration:",
    "Investigations:",
    "Procedures:",
    "Comorbidities:",
    "Risk Factors:",
    "Complications:",
]

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

TRAIN_EPOCHS = 3
TRAIN_BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 1024
SAVE_STEPS = 50
LOGGING_STEPS = 10
WARMUP_RATIO = 0.03
LR_SCHEDULER = "cosine"
OPTIM = "paged_adamw_8bit"

VALIDATION_SPLIT = 0.1
TEST_SPLIT = 0.1

DATASET_KEYS = {"clinical note", "structured clinical note"}

MAX_NEW_TOKENS = 300
TEMPERATURE = None
DO_SAMPLE = False

LOG_LEVEL = "INFO"

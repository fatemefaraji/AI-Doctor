import os
import logging
from typing import Dict, List, Optional
import uuid
import yaml
import torch
from unsloth import FastLanguageModel, is_bfloat16_supported
from transformers import TrainingArguments
from trl import SFTTrainer
from huggingface_hub import login
from datasets import load_dataset
import wandb
from dataclasses import dataclass
import pytest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('ai_doctor.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class for model and training parameters."""
    model_name: str = "dee/DeepSeek-R1-Distill-Llama-8B"
    max_sequence_length: int = 2048
    load_in_4bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: int = 1
    warmup_steps: int = 5
    max_steps: int = 60
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    output_dir: str = "outputs"
    dataset_name: str = "FreedomIntelligence/medical-o1-reasoning-SFT"
    dataset_split: str = "train[:500]"

    @classmethod
    def from_yaml(cls, file_path: str) -> 'Config':
        """Load configuration from a YAML file."""
        try:
            with open(file_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            return cls(**config_dict)
        except Exception as e:
            logger.error(f"Failed to load config from {file_path}: {e}")
            raise

class AIDoctor:
    """Main class for the AI Doctor system."""
    
    def __init__(self, config: Config, hf_token: str, wandb_token: str):
        self.config = config
        self.hf_token = hf_token
        self.wandb_token = wandb_token
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initialized AIDoctor with device: {self.device}")

    def setup_environment(self) -> None:
        """Set up Hugging Face and Weights & Biases."""
        try:
            login(self.hf_token)
            wandb.login(key=self.wandb_token)
            logger.info("Successfully logged into Hugging Face and Weights & Biases")
        except Exception as e:
            logger.error(f"Environment setup failed: {e}")
            raise

    def load_model(self) -> None:
        """Load the pretrained model and tokenizer."""
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.model_name,
                max_seq_length=self.config.max_sequence_length,
                dtype=None,
                load_in_4bit=self.config.load_in_4bit,
                token=self.hf_token
            )
            logger.info(f"Loaded model {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def setup_lora(self) -> None:
        """Apply LoRA configuration to the model."""
        try:
            self.model = FastLanguageModel.get_peft_model(
                model=self.model,
                r=self.config.lora_r,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                use_gradient_checkpointing="unsloth",
                random_state=3047,
                use_rslora=False,
                loftq_config=None
            )
            logger.info("Applied LoRA configuration")
        except Exception as e:
            logger.error(f"LoRA setup failed: {e}")
            raise

    def load_dataset(self) -> None:
        """Load and preprocess the medical dataset."""
        try:
            dataset = load_dataset(
                self.config.dataset_name,
                "en",
                split=self.config.dataset_split,
                trust_remote_code=True
            )
            self.finetune_dataset = dataset.map(self.preprocess_input_data, batched=True)
            logger.info("Loaded and preprocessed dataset")
        except Exception as e:
            logger.error(f"Dataset loading failed: {e}")
            raise

    @staticmethod
    def preprocess_input_data(examples: Dict) -> Dict:
        """Preprocess dataset for fine-tuning."""
        prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a medical expert with advanced knowledge in clinical reasoning, diagnostics, and treatment planning.
Please answer the following medical question.

### Question:
{}

### Response:
<think>
{}
</think>
{}"""
        inputs = examples["Question"]
        cots = examples["Complex_CoT"]
        outputs = examples["Response"]
        texts = [prompt_style.format(inp, cot, out) + "</s>" for inp, cot, out in zip(inputs, cots, outputs)]
        return {"texts": texts}

    def setup_trainer(self) -> None:
        """Set up the SFTTrainer for fine-tuning."""
        try:
            training_args = TrainingArguments(
                per_device_train_batch_size=self.config.batch_size,
                gradient_accumulation_steps=self.config.gradient_accumulation_steps,
                num_train_epochs=self.config.num_train_epochs,
                warmup_steps=self.config.warmup_steps,
                max_steps=self.config.max_steps,
                learning_rate=self.config.learning_rate,
                fp16=not is_bfloat16_supported(),
                bf16=is_bfloat16_supported(),
                logging_steps=10,
                optim="adamw_8bit",
                weight_decay=self.config.weight_decay,
                lr_scheduler_type="linear",
                seed=3407,
                output_dir=self.config.output_dir,
                report_to="wandb"
            )
            self.trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                train_dataset=self.finetune_dataset,
                dataset_text_field="texts",
                max_seq_length=self.config.max_sequence_length,
                dataset_num_proc=1,
                args=training_args
            )
            logger.info("Trainer setup complete")
        except Exception as e:
            logger.error(f"Trainer setup failed: {e}")
            raise

    def fine_tune(self) -> None:
        """Run the fine-tuning process."""
        try:
            with wandb.init(project="Fine-tune-AIDoctor", job_type="training", anonymous="allow"):
                trainer_stats = self.trainer.train()
                logger.info(f"Fine-tuning completed with stats: {trainer_stats}")
        except Exception as e:
            logger.error(f"Fine-tuning failed: {e}")
            raise

    def infer(self, questions: List[str], max_new_tokens: int = 1200) -> List[str]:
        """Run inference on a list of questions."""
        try:
            FastLanguageModel.for_inference(self.model)
            prompt_style = """Below is a task description along with additional context provided in the input section. Your goal is to provide a well-reasoned response that effectively addresses the request.

Before crafting your answer, take a moment to carefully analyze the question. Develop a clear, step-by-step thought process to ensure your response is both logical and accurate.

### Task:
You are a medical expert specializing in clinical reasoning, diagnostics, and treatment planning. Answer the medical question below using your advanced knowledge.

### Query:
{}

### Answer:
<think>{}
"""
            inputs = self.tokenizer(
                [prompt_style.format(q, "") for q in questions],
                return_tensors="pt",
                padding=True
            ).to(self.device)
            outputs = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )
            responses = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            return [r.split("### Answer:")[1].strip() for r in responses]
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise

# Unit Tests
def test_aidoctor():
    config = Config()
    ai_doctor = AIDoctor(config, os.getenv("HF_TOKEN", "dummy_token"), os.getenv("WANDB_API_TOKEN", "dummy_token"))
    
    # Test preprocessing
    sample_data = {
        "Question": ["What is the diagnosis?"],
        "Complex_CoT": ["Step 1: Analyze symptoms."],
        "Response": ["Likely flu."]
    }
    processed = ai_doctor.preprocess_input_data(sample_data)
    assert "texts" in processed
    assert len(processed["texts"]) == 1
    assert "</s>" in processed["texts"][0]
    
    # Test device detection
    assert ai_doctor.device in ["cuda", "cpu"]

if __name__ == "__main__":
    # Load configuration
    config = Config()  # In production, load from YAML file: Config.from_yaml("config.yaml")
    
    # Initialize AI Doctor
    hf_token = os.getenv("HF_TOKEN")
    wandb_token = os.getenv("WANDB_API_TOKEN")
    ai_doctor = AIDoctor(config, hf_token, wandb_token)
    
    # Setup and run
    ai_doctor.setup_environment()
    ai_doctor.load_model()
    ai_doctor.setup_lora()
    ai_doctor.load_dataset()
    ai_doctor.setup_trainer()
    ai_doctor.fine_tune()
    
    # Test inference
    questions = [
        """A 61-year-old woman with a long history of involuntary urine loss during activities like coughing or sneezing
           but no leakage at night undergoes a gynecological exam and Q-tip test. Based on these findings,
           what would cystometry most likely reveal about her residual volume and detrusor contractions?""",
        """A 59-year-old man presents with a fever, chills, night sweats, and generalized fatigue,
           and is found to have a 12 mm vegetation on the aortic valve. Blood cultures indicate gram-positive, catalase-negative,
           gamma-hemolytic cocci in chains that do not grow in a 6.5% NaCl medium.
           What is the most likely predisposing factor for this patient's condition?"""
    ]
    responses = ai_doctor.infer(questions)
    for q, r in zip(questions, responses):
        logger.info(f"Question: {q}\nResponse: {r}\n")
    
    # Run tests
    pytest.main(["-v", __file__])

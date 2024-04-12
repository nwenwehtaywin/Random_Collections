import argparse

class QGen_galaxie:

    import torch
    def __init__(self,GEN):
        self.GEN=GEN
        self.config={
            "output_dir":"./"+self.GEN.model,
            'save_model':self.GEN.model+"-chkpt"
        }

    def aceclerate_fn(self):
        from accelerate import FullyShardedDataParallelPlugin, Accelerator
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

        fsdp_plugin = FullyShardedDataParallelPlugin(
            state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
            optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
        )

        accelerator = Accelerator(fsdp_plugin=fsdp_plugin)
        return accelerator
    
    def load_dataset(self):
        from datasets import load_dataset
        db_id=self.GEN.database
        instruct_tune_dataset = load_dataset(db_id)
        instruct_tune_dataset = instruct_tune_dataset.filter(lambda x: x["source"] == "dolly_hhrlhf")
        return instruct_tune_dataset
    
    def create_prompt(sample):
        """
        Update the prompt template:
        Combine both the prompt and input into a single column.

        """
        bos_token = "<s>"
        original_system_message = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
        system_message = "Use the provided input to create an instruction that could have been used to generate the response with an LLM."
        response = sample["prompt"].replace(original_system_message, "").replace("\n\n### Instruction\n", "").replace("\n### Response\n", "").strip()
        input = sample["response"]
        eos_token = "</s>"

        full_prompt = ""
        full_prompt += bos_token
        full_prompt += "### Instruction:"
        full_prompt += "\n" + system_message
        full_prompt += "\n\n### Input:"
        full_prompt += "\n" + input
        full_prompt += "\n\n### Response:"
        full_prompt += "\n" + response
        full_prompt += eos_token

        return full_prompt
    
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training

        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.GEN.model,
            device_map='auto',
            quantization_config=nf4_config,
            use_cache=False
        )
        
        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            r=64,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, peft_config)
        return model,peft_config

    def load_tokenizer(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.GEN.model)

        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer
    
    def generate_response(self,prompt, tokenizer,model):

        encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
        model_inputs = encoded_input.to('cuda')
        generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)
        decoded_output = tokenizer.batch_decode(generated_ids)
        return decoded_output[0].replace(prompt, "")

    def fit_model(self,model,tokenizer,train_db,peft_config):
        from transformers import TrainingArguments
        from trl import SFTTrainer

        args = TrainingArguments(
            output_dir = self.config['output_dir'],
            num_train_epochs=self.GEN.epoch
            #max_steps = 100, # comment out this line if you want to train in epochs
            per_device_train_batch_size = 4,
            warmup_steps = 0.03,
            logging_steps=10,
            save_strategy="epoch",
            #evaluation_strategy="epoch",
            evaluation_strategy="steps",
            eval_steps=20, # comment out this line if you want to evaluate at the end of each epoch
            learning_rate=2e-4,
            bf16=True,
            lr_scheduler_type='constant',
        )

        
        max_seq_length = 2048

        trainer = SFTTrainer(
            model=model,
            peft_config=peft_config,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            packing=True,
            formatting_func=self.create_prompt, # this will aplly the create_prompt mapping to all training and test dataset
            args=args,
            train_dataset=train_db["train"],
            eval_dataset=train_db["test"]
        )
        trainer.train()
        return trainer

    def savemodel(self,trainer):
        trainer.save_model(self.config['save_model'])
        print("model is saved.")

    def load_model(self):
        merged_model = model.merge_and_unload()



if __name__=="__main__":
    parser=argparse.ArgumentParser()

    parser.add_argument('-model',
                        '--model',
                        type=bool,
                        default='google/flan-t5-base',
                        help="name of LLM")
    #"mistralai/Mistral-7B-Instruct-v0.1"
    parser.add_argument('-database',
                        '--database',
                        type=int,
                        default='mosaicml/instruct-v3',
                        help="name of db")
    parser.add_argument('-epoch',
                        '--epoch',
                        type=int,
                        default=3,
                        help="number of epoch")

    FLAGS = parser.parse_args()
    q_gen=QGen_galaxie(FLAGS)
    print("Loading dataset")
    print("*"*100)
    db=q_gen.load_dataset()
    print("loading tokenizer")
    print("*"*100)
    tokenizer=q_gen.load_tokenizer()
    print("Loading model")
    print("*"*100)
    model,config=q_gen.load_model()
    print("training........")
    print("*"*100)
    trainer=q_gen.fit_model(model,tokenizer,db,config)
    q_gen.savemodel(trainer)
    prompt=db['test'][100]['prompt']
    response=db['test'][100]['response']
    print("Generate the questions.......")
    print("*"*100)
    q_gen.generate_response(prompt,tokenizer, model)




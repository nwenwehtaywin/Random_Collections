class summarize_text:
    def __init__(self,config):
        self.config=config

    def load_data(self):
        from datasets import load_dataset
        billsum = load_dataset(self.config['db_id'], split="ca_test")
        #reduce db size
        billsum = billsum.train_test_split(test_size=0.2)
        return billsum
        
    def load_tokenizer(self):
        tokenizer=""
        if self.config['model_id']=='t5-small':
          from transformers import AutoTokenizer
          tokenizer = AutoTokenizer.from_pretrained(self.config['model_id'])
        elif self.config['model_id']=='facebook/bart-base':
          from transformers import BartTokenizer
          tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        return tokenizer

    def preprocess_function(self,examples):
        prefix = "summarize: "
        tokenizer=self.load_tokenizer()
        inputs = [prefix + doc for doc in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=1024, truncation=True)
    
        labels = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def token_fn(self,billsum):
        tokenized_billsum = billsum.map(self.preprocess_function, batched=True)
        return tokenized_billsum

    def load_model(self):
        model=""
        if self.config['model_id']=='t5-small':
            from transformers import AutoModelForSeq2SeqLM
            model = AutoModelForSeq2SeqLM.from_pretrained(self.config['model_id'])
        elif self.config['model_id']=='facebook/bart-base':
          from transformers import BartForConditionalGeneration
          model = BartForConditionalGeneration.from_pretrained("facebook/bart-large", forced_bos_token_id=0)
          
        return model

    def fit_model(self,tokenizer,model,tokenized_billsum):
        from transformers import DataCollatorForSeq2Seq
        from transformers import  Seq2SeqTrainingArguments, Seq2SeqTrainer
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config['output_dir'],
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=10,
            predict_with_generate=True,
            fp16=True,
            #push_to_hub=True,
            load_best_model_at_end=True,
            save_strategy = "epoch"
        )
    
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_billsum["train"],
            eval_dataset=tokenized_billsum["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
       
    def compute_metrics(self,eval_pred):
        import evaluate
        import numpy as np
        rouge = evaluate.load("rouge")
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
        result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)
    
        return {k: round(v, 4) for k, v in result.items()}
    
    def evaluate_model(self,tokenizer):
        text = """summarize: The Inflation Reduction Act lowers prescription drug costs, \
        health care costs, and energy costs. It's the most aggressive action on tackling the climate crisis in American history, \
        which will lift up American workers and create good-paying, union jobs across the country. It'll lower the deficit and ask \
        the ultra-wealthy and corporations to pay their fair share. And no one making under $400,000 per year will pay a penny more in taxes."""
        from transformers import pipeline,AutoModelForSeq2SeqLM
        loaded_model = AutoModelForSeq2SeqLM.from_pretrained(self.config['output_dir']+"/checkpoint-500")
        summarizer = pipeline("summarization", model=loaded_model,tokenizer=tokenizer)
        print(summarizer(text))

if __name__=="__main__":
    config={
        #"model_id":"t5-small", #-->worked well
        "model_id":"facebook/bart-base",
        "db_id":"billsum",
        "output_dir":"summarize-model-fb-bart",
    }
    summ=summarize_text(config)
    billsum=summ.load_data()
    tokenized_data=summ.token_fn(billsum)
    model=summ.load_model()
    tokenizer=summ.load_tokenizer()
    summ.fit_model(tokenizer,model,tokenized_data)
    
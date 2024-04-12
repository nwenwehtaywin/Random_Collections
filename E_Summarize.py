class SummarizeText:
    
    def __init__(self):
        import torch
        self.config={
            "model_id":"google/pegasus-cnn_dailymail",
            "database_id":"cnn_dailymail",
            "epoch":3,
            "max_input_length":200,
            "max_target_length":30,
            "device":'cuda' if torch.cuda.is_available() else 'cpu'
        }
        self.test_data=""

    def load_data(self):
        from datasets import load_dataset
        db=load_dataset(self.config['database_id'],'3.0.0')
        #db['train']=db['train'].shuffle(True).select(range(50000))
        #db['valid']=db['valididation'].shuffle(True).select(range(10000))
        self.test_data=db['test'].shuffle(True).select(range(5000))
        return db

    def show_samples(self,db,num_samples=3,seed=42):
        sample=db.shuffle(seed=seed).select(range(num_samples))
        for e in sample:
            print(f"\narticle: {e['article']}")
            print(f"highlights: {e['highlights']}")

    def load_tokenizer(self):
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(self.config['model_id'])
        return tokenizer
        
    def preprocess_function(self,examples):
        model_inputs=tokenizer(
            examples['article'],
            max_length=self.config['max_input_length'],
            truncation=True
        )
        labels=tokenizer(
            examples['highlights'],max_length=self.config['max_target_length'],truncation=True
        )
        
        model_inputs['labels']=labels['input_ids']
        return model_inputs
        
    def token_db(self,db):
        tokenized_db=db.map(self.preprocess_function,batched=True)
        tokenized_db=tokenized_db.remove_columns(db['train'].column_names)
        return tokenized_db

    def load_model(self):
        from transformers import AutoModelForSeq2SeqLM
        model=AutoModelForSeq2SeqLM.from_pretrained(self.config['model_id']).to(self.config['device'])
        return model

    def fit_model(self,tokenizer,model,tokenized_db):
        from transformers import TrainingArguments, Trainer
        from transformers import DataCollatorForSeq2Seq

        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        trainer_args = TrainingArguments(
            output_dir='my_summarize', 
            num_train_epochs=self.config['epoch'], 
            warmup_steps=100,
            per_device_train_batch_size=1, 
            per_device_eval_batch_size=4,
            weight_decay=0.01, 
            logging_steps=10,
            evaluation_strategy='steps', 
            eval_steps=100, 
            save_steps=1e6,
            gradient_accumulation_steps=16
        ) 
        trainer = Trainer(model=model, 
                          args=trainer_args,
                          tokenizer=tokenizer, 
                          data_collator=seq2seq_data_collator,
                          train_dataset=tokenized_db["train"], 
                          eval_dataset=tokenized_db["validation"])
        trainer.train()
        return trainer
    
    def generate_batch_sized_chunks(self,list_of_elements,batch_size):
        for i in range(0,len(list_of_elements),batch_size):
            yield list_of_elements[i:i+batch_size]

    def calculate_metric_on_test_data(self,dataset,metric,model,tokenizer,
                                 batch_size=16,
                                 column_text='article',
                                 column_summary='highlights'):
        article_batches=list(self.generate_batch_sized_chunks(dataset[column_text],batch_size))
        target_batches=list(self.generate_batch_sized_chunks(dataset[column_summary],batch_size))
        
        for article_batch,target_batch in tqdm(
            zip(article_batches,target_batches),total=len(article_batches)
            ):
            inputs=tokenizer(article_batch,max_length=self.config['max_input_length'],
                            truncation=True,padding='max_length',return_tensors='pt'
                            )
            summaries=model.generate(input_ids=inputs['input_ids'].to(self.config['device']),
                                    attention_mask=inputs['attention_mask'].to(self.config['device']),
                                     length_penalty=0.8,num_beams=8,max_length=128
                                    )
            decoded_summaries=[tokenizer.decode(s,skip_special_tokens=True,
                                               clean_up_tokenization_spaces=True
                                               ) for s in summaries]
            decoded_summaries=[d.replace(""," ") for d in decoded_summaries]
            
            metric.add_batch(predictions=decoded_summaries,references=target_batch)
        score=metric.compute()
        return score

    def show_test_result(self,db,trainer):
        from datasets import load_metric
        rouge_metric = load_metric('rouge')
        score = self.calculate_metric_on_test_ds(
            db['test'], rouge_metric, trainer.model, tokenizer, batch_size = 2, column_text = 'dialogue', column_summary= 'summary'
        )
        
        rouge_dict = dict((rn, score[rn].mid.fmeasure ) for rn in rouge_names )
        
        print(pd.DataFrame(rouge_dict, index = [f'pegasus'] ))


if __name__=="__main__":
    summ=SummarizeText()
    db=summ.load_data()
    #summ.show_samples(db['test'],3)
    tokenizer=summ.load_tokenizer()
    model=summ.load_model()
    tokenized_db=summ.token_db(db)
    
    trainer=summ.fit_model(tokenizer,model,tokenized_db)
    summ.show_test_result(db,trainer)

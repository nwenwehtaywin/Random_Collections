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
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.config['model_id'])
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
        return trainer
        
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

    def summarize_each_text(self,billsum,tok):
        from transformers import pipeline
        from random import randrange
        
        from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
        mdl=AutoModelForSeq2SeqLM.from_pretrained("./summarize-model/checkpoint-310")
        summarizer=pipeline('summarization',model=mdl,device='cuda',tokenizer=tok)
        
        # select a random test sample
        sample = billsum['test'][randrange(len(billsum["test"]))]
        print(f"article: \n{sample['text']}\n---------------")
        # summarize dialogue
        res = summarizer(sample["text"])
        print("...........Summarize Text................")
        print(f"flan-t5-base summary:\n{res[0]['summary_text']}")

    def generate_batch_sized_chunks(list_of_elements,batch_size):
            for i in range(0,len(list_of_elements),batch_size):
                yield list_of_elements[i:i+batch_size] 
                
    def calculate_metric_on_test_data(dataset,metric,model,tokenizer,
                                     batch_size=16,
                                     column_text='text',
                                     column_summary='summary'):
            from tqdm import tqdm
            
            article_batches=list(generate_batch_sized_chunks(dataset[column_text],batch_size))
            target_batches=list(generate_batch_sized_chunks(dataset[column_summary],batch_size))
            
            for article_batch,target_batch in tqdm(
                zip(article_batches,target_batches),total=len(article_batches)
                ):
                inputs=tokenizer(article_batch,max_length=100,
                                truncation=True,padding='max_length',return_tensors='pt'
                                )
                summaries=model.generate(input_ids=inputs['input_ids'].to('cuda'),
                                        attention_mask=inputs['attention_mask'].to('cuda'),
                                         length_penalty=0.8,num_beams=8,max_length=128
                                        )
                decoded_summaries=[tokenizer.decode(s,skip_special_tokens=True,
                                                   clean_up_tokenization_spaces=True
                                                   ) for s in summaries]
                decoded_summaries=[d.replace(""," ") for d in decoded_summaries]
                
                metric.add_batch(predictions=decoded_summaries,references=target_batch)
            score=metric.compute()
            return score
    
    def fun(db,trainer):
            from datasets import load_metric
            rouge_metric = load_metric('rouge')
            score = calculate_metric_on_test_data(
                db['test'], rouge_metric, trainer.model, tokenizer, batch_size = 2, column_text = 'text', column_summary= 'summary'
            )
            
            rouge_dict = dict((rn, score[rn].mid.fmeasure ) for rn in score )
            
            print(pd.DataFrame(rouge_dict, index = [f'pegasus'] ))

if __name__=="__main__":
    config={
        "model_id":"t5-small", 
        #google/pegasus-cnn_dailymail 
        #Falconsai/text_summarization
        #google/roberta2roberta_L-24_gigaword
        #cnicu/t5-small-booksum
        #google/pegasus-multi_news
        #philschmid/bart-large-cnn-samsum
        #facebook/bart-large-cnn --> very large
        
        "db_id":"billsum",
        "output_dir":"summarize-model",
        "max_input_length":200,
    }
    summ=summarize_text(config)
    billsum=summ.load_data()
    tokenized_data=summ.token_fn(billsum)
    model=summ.load_model()
    tokenizer=summ.load_tokenizer()
    trainer=summ.fit_model(tokenizer,model,tokenized_data)
    summ.summarize_each_text(billsum)  
    summ.show_test_result(billsum,trainer)
    summ.summarize_each_text(billsum,tokenizer)
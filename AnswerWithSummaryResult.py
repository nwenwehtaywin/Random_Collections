import openai
from IPython.core.display import display, HTML
import cohere
import numpy as np
openai.api_key = "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"

class SemiSearch_Cohere:
    def __init__(self,path):
        self.path=path

    def load_pdf(self):
        from llmsherpa.readers import LayoutPDFReader

        llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
        #pdf_url = "https://www.shinzen.org/wp-content/uploads/2016/08/WhatIsMindfulness_SY_Public_ver1.5.pdf" # also allowed is a file path e.g. /home/downloads/xyz.pdf\
        pdf_reader = LayoutPDFReader(llmsherpa_api_url)
        doc = pdf_reader.read_pdf(self.path)
        return doc

    def embed_docs(self,doc):
        cohere_key = "loiPbv1I8T5Ngz3vpFKPlgjNYzGUaHuj52iva1dt"
        co = cohere.Client(cohere_key)
        
        contexts = []
        for chunk in doc.chunks():
          contexts.append(chunk.to_context_text())
        
        #Encode your documents with input type 'search_document'
        doc_emb = co.embed(contexts, input_type="search_document", model="embed-english-v3.0").embeddings
        doc_emb = np.asarray(doc_emb)
        return co,doc_emb,contexts

    def ask(self,query,co,doc_emb,contexts):
      #Encode your query with input type 'search_query'
      query_emb = co.embed([query], input_type="search_query", model="embed-english-v3.0").embeddings
      query_emb = np.asarray(query_emb)
      query_emb.shape
    
      #Compute the dot product between query embedding and document embedding
      scores = np.dot(query_emb, doc_emb.T)[0]
    
      #Find the highest scores
      max_idx = np.argsort(-scores)
      most_relevant_contexts = []
      top_k = 10
    
      #Get only the top contexts to keep the context for openai small
      for idx in max_idx[0:top_k]:
        most_relevant_contexts.append(contexts[idx])
    
      #Call OpenAI to synthesize answers
      passages = "\n".join(most_relevant_contexts)
      prompt = f"Read the following passages and answer the question: {query}\n passages: {passages}"
      completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": prompt}])
      synthesized_answer = completion.choices[0].message.content
    
      print(f"Query: {query}")
      print(f"Answer: {synthesized_answer}")
      print("\nRelevant contexts: \n")
      for ctx in most_relevant_contexts:
          print(ctx)
          print("--------")

if __name__=="__main__":
    filepath='pdf/ai-assisted patent drafting demo.pdf'
    searchObj=SemiSearch_Cohere(filepath)
    doc=searchObj.load_pdf()
    co,doc_emb,contexts=searchObj.embed_docs(doc)

    question='What is the best cloud computing environment?'
    searchObj.ask(question,co,doc_emb,contexts)
    
    
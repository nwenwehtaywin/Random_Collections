from langchain.document_loaders import PDFPlumberLoader
from langchain.text_splitter import CharacterTextSplitter, TokenTextSplitter
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain import HuggingFacePipeline
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
import torch
from transformers import AutoTokenizer
import re
import os
class PdfSummarize:
    def __init__(self,config:dict = {}):
        self.config = config
        self.OPENAI_API_KEY='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'
        
    def summary_chain(self,system_about,instruction):
        """
        Creates retrieval qa chain using vectordb as retrivar and LLM to complete the prompt
        """
        from langchain.prompts import (
            ChatPromptTemplate,
            PromptTemplate,
            SystemMessagePromptTemplate,
            AIMessagePromptTemplate,
            HumanMessagePromptTemplate,
        )
        from langchain.schema import (
            AIMessage,
            HumanMessage,
            SystemMessage
        )
        
        from langchain.chat_models import ChatOpenAI
        from langchain.chains import LLMChain
        context_template=system_about
        system_message_prompt = SystemMessagePromptTemplate.from_template(context_template)

        human_template=instruction
        human_message_prompt = HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    template=human_template,
                    input_variables=["paper_content"],
                )
            )
        chat_prompt_template = ChatPromptTemplate.from_messages([system_message_prompt,
                                                                 human_message_prompt])
        chat = ChatOpenAI(model_name=self.config['llm'],
                          temperature=0.2,openai_api_key=self.OPENAI_API_KEY)
        summary_chain = LLMChain(llm=chat, prompt=chat_prompt_template)
        return summary_chain
            
    def  load_pdf(self):
        from PyPDF2 import PdfReader
        import tiktoken
        
        reader = PdfReader(self.config['pdf_path'])
        parts = []

        def visitor_body(text, cm, tm, fontDict, fontSize):
            y = tm[5]
            if y > 50 and y < 720:
                parts.append(text)
        
        for page in reader.pages:
            # page = reader.pages[3]
            page.extract_text(visitor_text=visitor_body)
        
        text_body = "".join(parts)
        return text_body

        
    def preprocess_text(self,text):
        import re

        def remove_citations(text):
            split_text = re.split(r'(\[\d+\].*?(?=\[\d+\]|$))', text, flags=re.DOTALL)
            no_citations = [chunk for i, chunk in enumerate(split_text) if i % 2 == 0]
            citations = [chunk for i, chunk in enumerate(split_text) if i % 2 != 0]
            return ''.join(no_citations), ''.join(citations)
        
        def remove_after_references(text: str) -> str:
            """Remove everything after a line containing 'References'."""
            lines = text.split('\n')
            for i, line in enumerate(lines):
                if 'References' in line:
                    return '\n'.join(lines[:i+1])
            return text

        text=remove_after_references(text)
        text=remove_citations(text)
        return text

    def summarize_run(selfm,chain,data):
        from langchain.callbacks import get_openai_callback
        import textwrap
        
        with get_openai_callback() as cb:
            output = chain.run(data)
            # print(f"Total Tokens: {cb.total_tokens}")
            # print(f"Prompt Tokens: {cb.prompt_tokens}")
            # print(f"Completion Tokens: {cb.completion_tokens}")
            # print(f"Total Cost (USD): ${cb.total_cost}")
            # print(f"Calc Total Cost (USD): ${(cb.prompt_tokens/1000)*0.003 + (cb.completion_tokens/1000)*0.004}")
            print(output)

if __name__=="__main__":

    #llm model
    #LLM_OPENAI_GPT35 = "gpt-3.5-turbo" #cannot run due to token limit exceed
    LLM_OPENAI_GPT35_16k='gpt-3.5-turbo-16k' #-->worked well
    #LLM_GPT3='gpt-3.5-turbo-1106' #worked well
    #LLM_GPT3='gpt-3.5-turbo-instruct' #--> cannot access
    #LLM_GPT3='gpt-3.5-turbo-0613' #--> cannot , token limit
    #LLM_GPT3='gpt-3.5-turbo-16k-0613' #--> worked well
    #LLM_GPT3='gpt-3.5-turbo-0301' #--> cannot, token limit

    
    config = {"persist_directory":None,
          "load_in_8bit":False,
          "llm":LLM_OPENAI_GPT35_16k,
          "pdf_path":"free-privacy-policy.pdf"
          }

    summ=PdfSummarize(config)
    data=summ.load_pdf()
    data=summ.preprocess_text(data)
    system_about="""You are a helpful AI Researcher that specializes in analysing papers.\
    Please use all your expertise to approach this task. Output your content in markdown format and include titles where relevant."""
    instructions="Please summarize this paper focusing the key important takeaways for each section. Expand the summary on methods so they can be clearly understood. \n\n PAPER: \n\n{paper_content}"
    sum_chain=summ.summary_chain(system_about,instructions)
    summ.summarize_run(sum_chain,data)

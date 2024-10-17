from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.chains import RetrievalQA

pdfreader = PdfReader('SAC Training Document.pdf')

# read text from pdf
from typing_extensions import Concatenate
raw_text = ''
for i, page in enumerate(pdfreader.pages):
    content = page.extract_text()
    if content:
        raw_text += content

# We need to split the text using Character Text Split such that it should not increse token size
text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

embeddings = HuggingFaceEmbeddings(model_name="thenlper/gte-large")
vector_store = FAISS.from_texts(texts, embedding=embeddings)

#Import Model
llm = LlamaCpp(
    streaming = True,
    model_path="zephyr-7b-alpha.Q8_0.gguf",
    temperature=0.75,
    top_p=1,
    verbose=True,
    n_ctx=4096
)

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(search_kwargs={"k": 2}))
query = "what is Predictive Analytics ?"
qa.run(query)

import pickle
import warnings
import time
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.llms import Ollama

'''
This code loads vectorstore.pkl in the folder "vectorstore". If there is no such file, you should run embedding.py first.
Then the retriever is created from the vectorstore, returning top-5 similarity search pages from the documents.
A set of queries are prepared, aim to test the response from the LLM (ollama3.2).
The output from this code is the responses generated from LLM, along with the related sources and response time.

Created by: Piyapart Buttamart
Created date: 20 November 2024
'''

warnings.filterwarnings("ignore", category=DeprecationWarning)

def format_docs(docs):
    text = ''
    for doc in docs:
        source = get_source_from_doc(doc)
        content = doc.page_content
        text += '\n\nSource: ' + source + '\n'+ content.strip() + '\n ----------'
    return text

def format_docs_truncate(docs):
    text = format_docs(docs)
    return text[:5000]

def get_source_from_doc(doc):
    pp = str(doc.metadata['page']+1)
    source = str(doc.metadata['source'])
    start = source.find('/')
    source = source[start+1:]
    text = source + ', page ' + pp
    return text

def get_source_from_list(docs):
    return "\n".join(get_source_from_doc(doc) for doc in docs)

def generate_response(query):
    response = ""
    start_generate_time = time.time()
    for chunk in rag_chain.stream(query):
        print(chunk, end="", flush=True)
        response += chunk
    finish_time = time.time()
    return response, finish_time-start_generate_time

# Load vectorstore object from a file
vector_file = 'vectorstore/vectorstore.pkl'
with open(vector_file, "rb") as f:
    vectorstore = pickle.load(f)
print("Vector store is loaded.")


# build a retriever from a vectorstore using its .as_retriever method
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
# return a list of k page (class langchain_core.documents.base.Document)

template = '''
The following context contain source (PDF file name and page) and content inside, which are relevant technical details for answering the user's question. \n
CONTENT:\n
{context} \n
Answer the following question concisely and include all the relevant sources that you use at the end of the response in a form <filename, page> because these are the only sources available for the users. If the materials are not relevant or complete enough to confidently answer the user’s question, your best response is “the materials do not appear to be sufficient to provide a good answer.” \n
QUESTION: \n
{question}
'''
prompt = ChatPromptTemplate.from_template(template)

llm = Ollama(
    model="llama3.2",
    temperature=0.7,   # Adjusts randomness
    top_k=40,          # Reduces the probability of generating nonsense. A higher value (e.g. 100) will give more diverse answers, while a lower value (e.g. 10) will be more conservative. (Default: 40)
    top_p=0.5,         # Works together with top-k. A higher value (e.g., 0.95) will lead to more diverse text, while a lower value (e.g., 0.5) will generate more focused and conservative text. (Default: 0.9)
    verbose=False,
    cache=True
)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()} # input: query, output: context (a formatted string of retrieved doc content) and question (the  query)
    | prompt # input: context and question, output: filled prompt
    | llm # input: filled prompt, output: string
    | StrOutputParser() # input: string, output: processed readable response
)

query = ['What is the role of the signal converter, signal processor and microprocessor in ZMQ?' ,
'What are the three meter types discussed in the manual, and how do they differ?',
'What are the possible housing types for the ZMQ meter?',
'What is the significance of the hardware configuration ID for the ZMQ meter?',
'How is the network frequency calculated by the ZMQ metering system?',
'What are the default measured quantities available in C.4 meters?',
'Which firmware versions are used for meters with the C.2 and C.4 software configurations?',
'How is reactive energy allocated to four quadrants in the ZMQ system?',
'What is the purpose of the MAP120 tool mentioned in the manual?',
'Describe the process for calculating apparent energy in ZMQ meters.',
'How does the ZMQ system handle frequency monitoring for error detection?',
'What standards or protocols does the ZMQ meter use for communication interfaces?',
'Can the ZMQ meter support custom configurations for harmonic distortion analysis?',
'What specific features make the C.7 configuration suitable for the Indian market?',
'How does the ZMQ meter accommodate changes in energy tariffs through its configuration?',
'What are the use cases for the additional power supply in ZMQ meters?',
'What does the manual suggest regarding the accuracy limitations of voltage dips?',
"How does the manual address cybersecurity considerations in the ZMQ meter's design?",
'What applications are best suited for using the power quality recorder in the ZMQ meter?',
'Does the ZMQ meter provide direct support for integration with smart grid systems?',
'What is the latest version of the iPhone?',
'How do solar panels convert sunlight into electricity?',
'What are the main features of Windows 11?',
'Who discovered the theory of relativity?',
'How does blockchain technology work?',
'What is the capital of Japan?',
'Explain the process of DNA replication.',
'What are the key features of Tesla electric cars?',
'What are the health benefits of a Mediterranean diet?',
'How does 5G technology differ from 4G?']
total_time = 0
for q in query:
    print(f"{q}\n")
    response, generate_time = generate_response(q)
    print("\n**********")

    retrieved_docs = retriever.invoke(q)
    sources =  get_source_from_list(retrieved_docs)

    output_text = f'''
    {q} \n
    {response}\n
    Retrieved source:
    {sources}\n
    Generate time: {generate_time:.2f} seconds\n
    **********
    '''
    with open('response.txt', 'a', encoding='utf-8') as file:
        file.write(output_text)
    total_time += generate_time
average_time = total_time / len(query)
print(f'Average time: {f:.2f} seconds')
with open('response.txt', 'a') as file:
    file.write('Average time: {f:.2f} seconds')

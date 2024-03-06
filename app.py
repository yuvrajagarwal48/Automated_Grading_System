from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os
import requests
import essay_grader
import asyncio



app = Flask(__name__)
CORS(app)  # Enable CORS for the Flask app

# OpenAI configuration
os.environ["OPENAI_API_KEY"] = "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"

# Define prompt template for question answering
template = '''
{text}

Grade these questions and answers and give a final score out of 10
give the score as output only nothing else in the format score'''

prompt = PromptTemplate(
    input_variables=['text'],
    template=template
)

async def predict_async(text):
    await asyncio.sleep(1)
    return essay_grader.predict(text)

# Initialize OpenAI and LLMChain
llm = OpenAI(temperature=0.6)
chain1 = LLMChain(llm=llm, prompt=prompt)

# Function to fetch quiz questions from the API
def fetch_quiz_questions(difficulty, topic):
    url = f"https://opentdb.com/api.php?amount=5&difficulty={difficulty}&category={topic}&type=multiple"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["results"]
    else:
        print("Failed to fetch quiz questions.")
        return []

# Function to fetch available categories from the API
def fetch_categories():
    url = "https://opentdb.com/api_category.php"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["trivia_categories"]
    else:
        print("Failed to fetch categories.")
        return []

# Flask API endpoint to fetch categories
@app.route('/categories', methods=['GET'])
def get_categories():
    categories = fetch_categories()
    return jsonify(categories)

# Flask API endpoint to fetch quiz questions
@app.route('/quiz', methods=['POST'])
def get_quiz():
    data = request.get_json()
    difficulty = data['difficulty']
    topic_id = data['topic_id']
    quiz_questions = fetch_quiz_questions(difficulty, topic_id)
    return jsonify(quiz_questions)

# Flask API endpoint to grade questions and answers
@app.route('/grade', methods=['POST'])
def grade():
    data = request.get_json()
    question = data['question']
    answer = data['answer']
    questions_answers = f'Q:"{question}"\nAnswer:{answer}'
    output = chain1.invoke(questions_answers)
    score = output['text']
    return jsonify({'score': score})

# Function to analyze PDF
def analyze_pdf(pdf):
    pdfreader = PdfReader(pdf)
    raw_text = ''
    for i, page in enumerate(pdfreader.pages):
        content = page.extract_text()
        if content:
            raw_text += content
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 800,
    chunk_overlap  = 200,
    length_function = len,
    )
    texts = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()

    document_search = FAISS.from_texts(texts, embeddings)

    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    query = "perform analysis on the provided question and answers and generate mistakes and areas of improvement"
    docs = document_search.similarity_search(query)
    analysis = chain.run(input_documents=docs, question=query)
    return analysis


@app.route('/analyze_pdf', methods=['POST'])
def analyze_pdf_endpoint():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    # Check if the file is a PDF
    if file.filename.endswith('.pdf'):
        analysis = analyze_pdf(file)
        return jsonify(analysis)
    else:
        return jsonify({'error': 'File is not a PDF'}), 400
    

    
@app.route('/predict', methods=['POST'])
async def predict_text():
    data = request.get_json()
    text = data['text']
    
    # Define the maximum length of text to process in each chunk
    max_chunk_length = 1000
    
    # Split the text into chunks
    chunks = [text[i:i+max_chunk_length] for i in range(0, len(text), max_chunk_length)]
    
    # Predict for each chunk asynchronously
    predictions = []
    for chunk in chunks:
        prediction = await predict_async(chunk)
        predictions.append(prediction)
    
    return jsonify({'predictions': str(predictions)})


if __name__ == '__main__':
    app.run(debug=True)
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")

## Prompt Template
quiz_analysis_prompt = """

Evaluate the student's understanding and application of concepts through a detailed analysis of their responses to a quiz. 
Compile the analysis results into a structured JSON format for easy interpretation and integration.

Language Requirement:

The detected language from the initial quiz must be consistently used throughout the entire analysis and result.
All content in the result should be in the detected language. No other language should be used.
Analysis Guidelines:

For Each Question: use only detected language translate if needed 

Question Type and Content Description
Clearly describe the type of question and the content it covers.
Correct Answer
Provide the correct answer based on the quiz key.
Student's Answer
Include the student's actual response.
Identify for Each Question:

Cognitive Level Based on Bloom's Taxonomy
Determine the cognitive level: Knowledge, Comprehension, Application, Analysis, Synthesis/Creation, Evaluation.
Achievement Level of the Student's Response
Categorize as Achieved, Partially Achieved, or Not Achieved.
Potential Reasons for the Student's Performance
Identify possible reasons for the student's performance, such as misunderstandings, instructional gaps, etc.
Feedback and Strategies for Improvement
Provide constructive feedback and tailored strategies for improvement based on the identified cognitive level.
Scoring Each Response:

1-3 (Not Achieved): Misunderstands content, lacks required cognitive skill, incorrect or irrelevant answers.
4-6 (Partially Achieved): Some understanding but incomplete, partial correctness.
7-9 (Mostly Achieved): Largely correct, good understanding with minor errors.
10 (Fully Achieved): Completely meets cognitive demands, excellent understanding and thorough response.
Create JSON Object:

Include the score and cognitive level for each question.
Ensure the result matches the language detected from the initial quiz.


before providing the result , ask you self did you use any other language , if yes you used two languages always refer back to the initial quiz language

"""
prompt=ChatPromptTemplate.from_messages(
    [
        ("system",quiz_analysis_prompt),
        ("user","Question:{question}")
    ]
)
## streamlit framework

st.title('Langchain Demo With Phi-2 API')
input_text=st.text_input("Search the topic u want")

# ollama LLAma2 LLm 
llm=Ollama(model="phi")
output_parser=StrOutputParser()
chain=prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))
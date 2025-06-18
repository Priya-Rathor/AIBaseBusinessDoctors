# %%
from langgraph.graph import Graph
from pydantic import BaseModel
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from langchain_community.tools import TavilySearchResults
from typing import Optional, TypedDict
from langgraph.graph import END, START, StateGraph, MessagesState
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from typing import Union


from fastapi import FastAPI
load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
os.environ["TAVILY_API_KEY"]=os.getenv("TAVILY_API_KEY")
llm=ChatGroq(model="llama3-groq-70b-8192-tool-use-preview")


# %%
class InputState(TypedDict):
    user_question:str
    user_answer:str
    gen_context:str
    question_type:str
    subject: Optional[str]
class OutputState(TypedDict):
    feedback:str
    Total_Score:int  
    Accuracy: int
    Relevance_to_the_Question:int
    percentage:float
    Logical_Consistency: int
    Scientific_Accuracy: int
    Depth_and_Explanation: int
    Factual_Correctness: int
class OverallState(InputState,OutputState):
    pass


# %%
tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
    
)
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

# %%
import datetime

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig, chain

today = datetime.datetime.today().strftime("%D")
prompt = ChatPromptTemplate(
    [
        ("system", f"You are a helpful assistant. The date today is {today}. If the user's question requires external data or search, call a tool to fetch that data."),
        ("human", "{user_input}"),
        ("placeholder", "{messages}"),
    ]
)

llm_with_tools = llm.bind_tools([tool,wikipedia])

llm_chain = prompt | llm_with_tools


@chain
def tool_chain(state:InputState):
    user_input=state["user_question"]
    input_ = {"user_input": user_input}
    ai_msg = llm_chain.invoke(input_)
    print("ai_msg",ai_msg)
    tool_msgs = tool.batch(ai_msg.tool_calls)
    print("tool_msgs",tool_msgs)
    z=llm_chain.invoke({**input_, "messages": [ai_msg, *tool_msgs]})
    print("toolchain",z)
    return {"gen_context":z}

class subject(BaseModel):
    subject:str
class question_re(BaseModel):
    reformatted_question:str
    
def question_reformat(state: InputState)->InputState:
    if state["user_question"]!="MCQ":
        prompt = f"""
    You are an assistant that helps to reformat questions. Your task is to rephrase the given question in a clear, structured, and grammatically correct way.

    ### Input:
    - **Original Question**: "{state['user_question']}"

    ### Task:
    Please reformat the question to make it more clear, direct, and formal if necessary. Your output should be a reformatted version of the original question.

    ### Output Format:
    Provide the reformatted question in the following JSON format:
    ```json
    {{
        "reformatted_question": "<Rephrased version of the input question>"
    }}
    ```

    """
    # Invoke the LLM with the prompt
        response_question = llm.invoke(prompt)
        try:
            parsed_response = question_re.parse_raw(response_question.content)  # Parse the JSON response
            print("Parsed subject:", parsed_response.reformatted_question)  # Print parsed subject
            return {"user_question":parsed_response.reformatted_question}
        except Exception as e:
            print(f"Error parsing response: {e}")
            return {"user_question":state["user_question"]}


def subject_checker(state: InputState)->InputState:
    prompt = f"""
You are an intelligent assistant trained to classify questions into specific subject categories. Your task is to analyze the given question and determine the most appropriate subject type based on its content.

### Input:
- **Question**: "{state['user_question']}"

### Subject Categories:
1. **Logical Reasoning**: Questions that involve patterns, puzzles, reasoning, or problem-solving.
- Examples: "If all cats are animals, and some animals are wild, can all cats be wild?" or "What comes next in the sequence: 2, 4, 8, 16, ...?"

2. **Science**: Questions that relate to scientific concepts, theories, or facts from subjects like Physics, Biology, Chemistry, etc.
- Examples: "Why does water boil at 100 degrees Celsius?" or "What is the role of mitochondria in a cell?"

3. **General Knowledge**: Questions that involve facts, history, geography, or any general topic outside of logic and science.
- Examples: "Who is the President of the United States in 2024?" or "Where is the Eiffel Tower located?"

### Task:
Analyze the question and classify it into one of the three categories: Logical Reasoning, Science, or General Knowledge.

### Output Format:
Provide your classification in the following JSON format:
```json
{{
    "subject":"Logical Reasoning / Science / General Knowledge"
}}
"""
    response_subject = llm.invoke(prompt)
    try:
        parsed_response = subject.parse_raw(response_subject.content)  # Parse the JSON response
        print("Parsed subject:", parsed_response.subject)  # Print parsed subject     
        return {"subject":parsed_response.subject}
    except Exception as e:
        print(f"Error parsing response: {e}")
        state["subject"] = "Unknown"




# %%


# Base Evaluation Output
class BaseEvaluationOutput(BaseModel):
    Total_Score: int
    feedback: str
    percentage: float

# Logical Evaluation Output
class LogicalEvaluationOutput(BaseEvaluationOutput):
    Logical_Consistency: int

# Science Evaluation Output
class ScienceEvaluationOutput(BaseEvaluationOutput):
    Scientific_Accuracy: int
    Depth_and_Explanation: int

# General Knowledge Evaluation Output
class GeneralKnowledgeEvaluationOutput(BaseEvaluationOutput):
    Factual_Correctness: int




# %%
def evaluate(state: InputState):
    """Evaluate the question and answer using the generated context with a focus on mathematical accuracy."""
    print("states",state)
    if state["question_type"] == "MCQ":
        print("question is",state["user_question"])
        prompt = f"""

    You are an expert evaluator specializing in {state["subject"]}. 
    Your task is to evaluate multiple-choice questions (MCQs). 
    Assign a score of 0 for incorrect answers and 1 for correct answer in Total_score.
    - **Question**: "{state['user_question']}"
    - **User's Answer**: "{state['user_answer']}"
    - **Question Type**: "{state['question_type']}"
    - **Reference Answer**: "{state['gen_context']}" (if available; otherwise, base your evaluation on logical principles)


    ### Output Format:
Provide your evaluation strictly in the following JSON format:
```json
{{
    "Total_Score": (integer 0 if the answer is wrong 1 if the answer is correct),
    "percentage": (float rounded to two decimal places),
    "feedback": "Constructive feedback focusing on scientific accuracy and depth."
}}
    """
        response_model=BaseEvaluationOutput
    elif state["subject"]=="Logical Reasoning":
        prompt = f"""
You are an expert evaluator specializing in logical reasoning questions. Your task is to evaluate the user's answer based on the following input details:

- **Question**: "{state['user_question']}"
- **User's Answer**: "{state['user_answer']}"
- **Reference Answer**: "{state['gen_context']}" (if available; otherwise, base your evaluation on logical principles)
- **Question Type**: "{state['question_type']}"

### Evaluation Criteria:
1. **Logical Consistency (10 points)**: Is the user's reasoning consistent and free of logical fallacies?
   - Assign full points if the reasoning is flawless.
   - Deduct points for errors, omissions, or contradictions.
2. check weather the user'answer is upto the point such that it cannot be solved completely. and also check if trying some shortcuts . if the answer doesn't match remove some marks
3. check for all possibilites of end answers which give end result.

### Scoring Instructions:
- Provide a score for each criterion out of 10.
- Calculate the total score out of 20 by summing the individual scores.
- Calculate the percentage as `(Total Score / 20) * 100` and round it to two decimal places.

### Feedback:
Provide constructive feedback that highlights strengths and areas of improvement in the user's reasoning.

### Output Format:
Provide your evaluation strictly in the following JSON format:
```json
{{
    "Logical_Consistency": (integer between 0-10),
    "Total_Score": (integer between 0-20),
    "percentage": (float rounded to two decimal places),
    "feedback": "Constructive feedback focusing on logical reasoning."
}}
"""
        response_model = LogicalEvaluationOutput

    elif state['subject'] in ['Science', 'Physics', 'Biology']:
        prompt = f"""
You are an expert evaluator specializing in scientific subjects. Your task is to evaluate the user's answer to a scientific question based on the following input details:

- **Question**: "{state['user_question']}"
- **User's Answer**: "{state['user_answer']}"
- **Reference Answer**: "{state['gen_context']}" (if available; otherwise, use scientific principles to evaluate)
- **Question Type**: "{state['question_type']}"

### Evaluation Criteria:
1. **Scientific Accuracy (10 points)**: Does the user's answer align with established scientific principles?
   - Assign full points for scientifically accurate and well-explained answers.
   - Deduct points for inaccuracies or omissions.
2. **Depth and Explanation (10 points)**: Does the user's answer demonstrate a good understanding of the topic and provide sufficient detail?
   - Assign full points for thorough and well-articulated answers.
   - Deduct points for incomplete or poorly explained answers.
3. check if the users answer is incomplete or not upto the final point where it has to be check the user is not trying anyshortcuts
### Scoring Instructions:
- Provide a score for each criterion out of 10.
- Calculate the total score out of 20 by summing the individual scores.
- Calculate the percentage as `(Total Score / 20) * 100` and round it to two decimal places.

### Feedback:
Provide constructive feedback that highlights strengths and areas of improvement, such as missing scientific details or logical inconsistencies.

### Output Format:
Provide your evaluation strictly in the following JSON format:
```json
{{
    "Scientific_Accuracy": (integer between 0-10),
    "Depth_and_Explanation": (integer between 0-10),
    "Total_Score": (integer between 0-20),
    "percentage": (float rounded to two decimal places),
    "feedback": "Constructive feedback focusing on scientific accuracy and depth."
}}
"""
        response_model = ScienceEvaluationOutput

    else:
        prompt = f"""
You are an expert evaluator specializing in general knowledge and miscellaneous subjects. Your task is to evaluate the user's answer based on the following input details:

- **Question**: "{state['user_question']}"
- **User's Answer**: "{state['user_answer']}"
- **Reference Answer**: "{state['gen_context']}" (if available; otherwise, base your evaluation on standard knowledge)
- **Question Type**: "{state['question_type']}"

### Evaluation Criteria:
1. **Factual Correctness (10 points)**: Is the user's answer factually accurate?
   - Assign full points for correct and precise answers.
   - Deduct points for inaccuracies or incorrect information.
2. check if the users answer is incomplete or not upto the final point where it has to be check the user is not trying anyshortcuts


### Scoring Instructions:
- Provide a score for each criterion out of 10.
- Calculate the total score out of 20 by summing the individual scores.
- Calculate the percentage as `(Total Score / 20) * 100` and round it to two decimal places.

### Feedback:
Provide constructive feedback that highlights areas of improvement, such as factual errors or lack of clarity.

### Output Format:
Provide your evaluation strictly in the following JSON format:
```json
{{
    "Factual_Correctness": (integer between 0-10),
    "Total_Score": (integer between 0-20),
    "percentage": (float rounded to two decimal places),
    "feedback": "Constructive feedback focusing on factual correctness and relevance."
}}
"""
        response_model = GeneralKnowledgeEvaluationOutput

    
    response = llm.invoke(prompt)
    print("response",response)
    try:
        parsed_response = response_model.parse_raw(response.content)
        print(parsed_response)
        return parsed_response.dict()
    except Exception as e:
        print("Error parsing response:", e)
        return {
            "marks": 0,
            "feedback": "Unable to parse response. Please ensure the format is correct.",
        }


# %%
builder=StateGraph(OverallState, input=InputState, output=OutputState)
builder.add_node("tailvy_tool",tool_chain)
builder.add_node("evalulate",evaluate)
builder.add_node("question_reformat",question_reformat)
builder.add_node("subject_finder",subject_checker)
builder.add_edge(START,"question_reformat")
builder.add_edge("question_reformat","subject_finder")
builder.add_edge("subject_finder","tailvy_tool")
builder.add_edge("tailvy_tool","evalulate")
builder.add_edge("evalulate",END)
graph=builder.compile()

# # %%
# from IPython.display import Image, display

# try:
#     display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     pass

# %%
# data=graph.invoke({"user_question":"who is the founder of apple company?","user_answer":"steve jobs ","gen_context:str":"None","question_type":"Short"})

# %%
# data

# %%
import nest_asyncio
import uvicorn
import subprocess
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException, status


# %%

app = FastAPI()
class Item(BaseModel):
    user_question: str
    user_answer: str
    question_type: str =None

@app.post("/evaluate")
def evaluate(item: Item):
    try:
        # Simulating the invocation of your external graph or model.
        data = graph.invoke({"user_question": item.user_question, "user_answer": item.user_answer, "gen_context": "None", "question_type": item.question_type})

        # Return the response with the obtained data
        return JSONResponse(
            content={"message": "Item evaluated successfully", "data": data},
            status_code=status.HTTP_200_OK  # HTTP 201 Created
        )
    except Exception as e:
        # Catch any exception and return an error response with a 400 or 500 status code
        return JSONResponse(
            content={"message": "An error occurred during evaluation", "error": str(e)},
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR  # HTTP 500 Internal Server Error
        )

@app.get("/hello")
def create_item():
    # Example of using the status code for 'Created' (201)
    
    return JSONResponse(
        content={"message": "Hello World"},
        status_code=status.HTTP_201_CREATED  # HTTP 201 Created
    )





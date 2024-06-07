import os
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = FastAPI()

script_path = os.getcwd() 
tokenizer_path = os.path.join(script_path, "tokenizer")
model_path = os.path.join(script_path, "model")

tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

class Question(BaseModel):
    question: str = Field(..., example="What criteria determine a route's eligibility for return?")

class Answer(BaseModel):
    answer: str

@app.post("/predict", response_model=Answer)
async def predict(question: Question = Body(..., example={"question": "Under what conditions can a route be returned?"})):
    try:
        input_ids = tokenizer.encode(question.question, return_tensors="pt")
        output = model.generate(input_ids, max_length=50, num_return_sequences=1)
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error predicting the answer: {str(e)}")
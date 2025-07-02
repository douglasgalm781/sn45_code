from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import submission

app = FastAPI()

# Initialize the LLM class from submission.py

swe_instance = submission.SWE()


class CallRequest(BaseModel):
    repo_location: str
    issue_description: str


@app.post("/call")
async def call_swe(request: CallRequest) -> dict:
    try:
        # Run the LLM object with the given inputs
        result = swe_instance(request.repo_location, request.issue_description)
        return {"result": result.model_dump()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=3000)

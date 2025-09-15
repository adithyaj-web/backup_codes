from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncio
import json
import sys
import os
import logging 
from app2 import init_provider

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the current directory to Python path to import agent
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agent import run_agent
from agent_codegen_updated import run_codegen_agent
from app import (
    get_llm_provider_for_project,
    get_openai_api_key_from_api,
    get_anthropic_api_key_from_api,
    client, anthropic_client, llm_provider
)
from openai import OpenAI
import anthropic
from cleaning_agent import normalize_payload
app = FastAPI(title="Test Data Generation Agent API")
class CleaningRequest(BaseModel):
    testCases: list  # Array of test case objects
    projectId: int
    imageFolder: str = "test_images"
    mode: str = "patch"

#@app.post("/agent/clean-testcases")
#async def clean_testcases(request: CleaningRequest):
#    """Clean test cases using the cleaning agent"""
#    try:
#        
#        cleaned_testcases = []
#        
#        for testcase in request.testCases:
#            cleaned_tc = normalize_payload(
#                raw=testcase,
#                image_folder=request.imageFolder,
#                mode=request.mode
#            )
#            cleaned_testcases.append(cleaned_tc)
#        
#        return {
#            "success": True, 
#            "data": {
#                "cleanedTestCases": cleaned_testcases,
#                "totalProcessed": len(cleaned_testcases)
#            }
#        }
#    except Exception as e:
#        logger.error(f"Error in cleaning agent: {str(e)}")
#        raise HTTPException(status_code=500, detail=str(e))

@app.post("/agent/clean-testcases")
async def clean_testcases(request: CleaningRequest):
    """Clean test cases using the cleaning agent"""
    try:
        # Import the cleaning agent
        from cleaning_agent import normalize_payload
        
        logger.info(f"[CLEANING] Received {len(request.testCases)} test cases for cleaning")
        logger.info(f"[CLEANING] Project ID: {request.projectId}")
        logger.info(f"[CLEANING] Image folder: {request.imageFolder}")
        
        cleaned_testcases = []
        
        for i, testcase in enumerate(request.testCases):
            logger.info(f"[CLEANING] Processing test case {i+1}/{len(request.testCases)}")
            logger.info(f"[CLEANING] Original test case structure: {list(testcase.keys())}")
            logger.info(f"[CLEANING] Original test case: {json.dumps(testcase, indent=2)}")
            
            # Add projectId to the test case data since the cleaning agent expects it there
            testcase_with_project_id = {**testcase, "projectId": request.projectId}
            
            cleaned_tc = normalize_payload(
                raw=testcase_with_project_id,
                image_folder=request.imageFolder,
                mode=request.mode
            )
            
            logger.info(f"[CLEANING] Cleaned test case: {json.dumps(cleaned_tc, indent=2)}")
            cleaned_testcases.append(cleaned_tc)
        
        logger.info(f"[CLEANING] Successfully cleaned all {len(cleaned_testcases)} test cases")
        
        return {
            "success": True, 
            "data": {
                "cleanedTestCases": cleaned_testcases,
                "totalProcessed": len(cleaned_testcases)
            }
        }
    except Exception as e:
        logger.error(f"[CLEANING] Error in cleaning agent: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    
class TestCaseRequest(BaseModel):
    testCase: dict

@app.post("/agent/run")
async def execute_agent(request: TestCaseRequest):
    """Execute the test data generation agent"""
    logger.info(f"Received request for test case: {request.testCase.get('tcId', 'unknown')}")
    try:
        test_case_json = json.dumps(request.testCase)
        result = await run_agent(test_case_json)
        logger.info(f"Agent returned: {len(result.get('TestData', []))} test data entries")
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        logger.info("Request completed successfully")    
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
# Add this new model class

class CodegenRequest(BaseModel):
    testCaseData: dict

# Add this new endpoint
@app.post("/agent/codegen")
async def execute_codegen_agent(request: CodegenRequest):
    """Execute the ATS code generation agent"""
    try:
        # Import the codegen agent function
        test_case_json = json.dumps(request.testCaseData)
        
        project_id = request.testCaseData.get("projectId") or request.testCaseData.get("testData", {}).get("projectId")
        if not project_id:
            raise HTTPException(status_code=400, detail="projectId is required in testCaseData")
        init_provider(int(project_id))
        global llm_provider, client, anthropic_client
        provider_name = get_llm_provider_for_project(project_id)
        if provider_name == "OpenAI":
            openai_key = get_openai_api_key_from_api()
            llm_provider = "openai"
            client = OpenAI(api_key=openai_key)
            anthropic_client = None
        elif provider_name == "Anthropic":
            anthropic_key = get_anthropic_api_key_from_api()
            llm_provider = "anthropic"
            anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
            client = None
        else:
            raise HTTPException(status_code=500, detail=f"Unknown provider name: {provider_name}")
        

        result = await run_codegen_agent(test_case_json)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["error"])
            
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7007)

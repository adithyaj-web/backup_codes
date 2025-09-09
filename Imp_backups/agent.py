#!/usr/bin/env python3
import asyncio
import json
import os
import sys
import re
import random
import uuid
from faker import Faker
from typing import Any, Dict, List, Optional
from datetime import datetime, timedelta
from langchain.memory import ConversationBufferMemory
from openai import OpenAI
from websockets.client import connect as ws_connect  # Note: Deprecation warning, but keep for now
from dotenv import load_dotenv 
from dataclasses import dataclass
import requests
import logging
import anthropic
load_dotenv() 

#USE_ANTHROPIC_CLAUDE = True
llm_provider: str = ""
client = None
anthropic_client = None
# ---------------------------------------------------------------------------
# DEBUGGING PRINT
# ---------------------------------------------------------------------------
print("DEBUG: Script started. Attempting initial setup.", file=sys.stderr)
sys.stderr.flush()

# ---------------------------------------------------------------------------
# GET PROVIDER NAME FOR THE API KEY SELECTION 
# ---------------------------------------------------------------------------

def get_llm_provider_for_project(project_id: int) -> Optional[str]:
    """Fetch LLM provider name for the given project ID from the API."""
    url = "http://localhost:3000/api/getLlmProviderForProjectId"
    payload = {"projectId": project_id}
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()
        provider_name = data.get("data")
        print(f"\n[INFO] LLM Provider for project {project_id}: {provider_name}\n", file=sys.stderr)
        return provider_name
    except Exception as e:
        print(f"[ERROR] Could not fetch LLM provider for project {project_id}: {e}", file=sys.stderr)
        return None

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)

def get_openai_api_key_from_api() -> str:
    """Fetch OpenAI API key from the API endpoint"""
    try:
        logger.info("[CONFIG] Fetching OpenAI API key from API endpoint")
        
        response = requests.get(
            "http://localhost:3000/api/secrets/OPENAI_API_KEY",
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        data = response.json()
        
        openai_key = data['value']
        logger.info(f"[CONFIG] OpenAI API Key loaded (length: {len(openai_key)} chars)")
        print(f"[CONFIG] OpenAI API Key loaded (length: {len(openai_key)} chars)", file=sys.stderr)
        return openai_key
        
    except requests.RequestException as e:
        logger.error(f"[CONFIG] Failed to fetch OpenAI API key from API: {e}")
        raise
    except Exception as e:
        logger.error(f"[CONFIG] Error processing OpenAI API key response: {e}")
        raise

def get_anthropic_api_key_from_api() -> str:
    """Fetch Anthropic API key from the API endpoint"""
    try:
        print(f"[CONFIG] Fetching Anthropic API key from API endpoint", file=sys.stderr)
        
        response = requests.get(
            "http://localhost:3000/api/secrets/ANTHROPIC_API_KEY",
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        data = response.json()
        
        anthropic_key = data['value']
        print(f"[CONFIG] Anthropic API Key loaded (length: {len(anthropic_key)} chars)", file=sys.stderr)
        print(f"[CONFIG] Anthropic API Key: {anthropic_key}", file=sys.stderr)
        
        return anthropic_key
        
    except requests.RequestException as e:
        print(f"[CONFIG] Failed to fetch Anthropic API key from API: {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f"[CONFIG] Error processing Anthropic API key response: {e}", file=sys.stderr)
        raise

#OPENAI_API_KEY = get_openai_api_key_from_api()
#ANTHROPIC_API_KEY = get_anthropic_api_key_from_api() if USE_ANTHROPIC_CLAUDE else None

MODEL_NAME = "gpt-4o-mini"  # GPT-4o-mini model from OpenAI
MCP_URL = (
    os.getenv("MCP_WS_URL")          # primary (works with compose + .env)
    or os.getenv("MCP_SERVER_URL")   # backward compatibility
    or "ws://localhost:8931"         # fallback for bare local dev
)
print("MCP_URL: ", MCP_URL, file=sys.stderr)

# Initialize OpenAI client
#client = OpenAI(api_key=OPENAI_API_KEY)
#anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY) if USE_ANTHROPIC_CLAUDE else None
# ---------------------------------------------------------------------------
# GLOBAL STATE
# ---------------------------------------------------------------------------
TEST_CASES: List[Dict[str, Any]] = []
DB_SCHEMA: Dict[str, Any] = {}
STATE: Dict[str, Any] = {}
CURRENT_ENVIRONMENT_ID: Optional[int] = None
AGENT_ID: str = ""
AGENT_SECRET_KEY: str = ""

# JWT token storage (will be set after authentication)
CURRENT_JWT_TOKEN: str = ""

# ADD THIS: Enhanced WebSocket message handling for token refresh
async def handle_websocket_message(ws, message_data):
    """Handle incoming WebSocket messages including token refresh"""
    global CURRENT_JWT_TOKEN
    
    try:
        if isinstance(message_data, str):
            message = json.loads(message_data)
        else:
            message = message_data
            
        # Handle token refresh notifications
        if message.get("method") == "token_refresh":
            params = message.get("params", {})
            new_token = params.get("newToken")
            expires_in = params.get("expiresIn")
            refreshed_at = params.get("refreshedAt")
            
            if new_token:
                CURRENT_JWT_TOKEN = new_token
                print(f"[JWT] Token refreshed automatically at {refreshed_at}", file=sys.stderr)
                print(f"[JWT] New token expires in: {expires_in}", file=sys.stderr)
                print(f"[JWT] Updated stored token (length: {len(CURRENT_JWT_TOKEN)} chars)", file=sys.stderr)
                print(f"[JWT] New JWT Token: {CURRENT_JWT_TOKEN}", file=sys.stderr)
            return True
            
        return False # Not a token refresh message
        
    except json.JSONDecodeError as e:
        print(f"[JWT] Failed to parse WebSocket message: {e}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"[JWT] Error handling WebSocket message: {e}", file=sys.stderr)
        return False
    
# ---------------------------------------------------------------------------
# AWS SECRETS MANAGER UTILITIES
# ---------------------------------------------------------------------------
async def fetch_credentials_from_api() -> tuple[str, str]:
    """Fetch agent credentials from the API endpoint"""
    try:
        print(f" Fetching credentials from API endpoints", file=sys.stderr)
        
        # Fetch agent ID
        agent_id_response = requests.get(
            "http://localhost:3000/api/secrets/AGENT_ID",
            headers={'Content-Type': 'application/json'}
        )
        agent_id_response.raise_for_status()
        agent_id_data = agent_id_response.json()
        
        print(f" Agent ID API Response: {agent_id_data}", file=sys.stderr)
        
        # Fetch secret key
        secret_key_response = requests.get(
            "http://localhost:3000/api/secrets/AGENT_SECRET_KEY", 
            headers={'Content-Type': 'application/json'}
        )
        secret_key_response.raise_for_status()
        secret_key_data = secret_key_response.json()
        
        print(f" Secret Key API Response: {secret_key_data}", file=sys.stderr)
        
        agent_id = agent_id_data['value']
        secret_key = secret_key_data['value']
        
        print(f" Credentials fetched successfully:", file=sys.stderr)
        print(f"   Agent ID: {agent_id}", file=sys.stderr)
        print(f"   Secret Key: {secret_key[:20]}...", file=sys.stderr)
        
        return agent_id, secret_key
        
    except requests.RequestException as e:
        print(f" Failed to fetch credentials from API: {e}", file=sys.stderr)
        raise
    except Exception as e:
        print(f" Error processing credentials response: {e}", file=sys.stderr)
        raise
# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------
def safe_dump(obj: Any) -> str:
    """Safely converts any Python object to a JSON string (or fallback to repr if it fails)."""
    try:
        return json.dumps(obj, indent=2)
    except Exception:
        return repr(obj)

def print_raw_response(tool_name: str, resp: Any, req_id: str):
    """Prints a clean version of the raw tool response to stderr for debugging"""
    #print(f"\n← RESPONSE [{tool_name}] ({req_id}) ←", file=sys.stderr)
    #print(safe_dump(resp), file=sys.stderr)
    sys.stderr.flush()

# ---------------------------------------------------------------------------
# OPENAI UTILITIES
# ---------------------------------------------------------------------------
#async def ask_openai(prompt: str, timeout: int = 60, temperature: float = 0.0) -> str:
#    """Send prompt to OpenAI GPT-4o-mini and get response text."""
#    print(f"Sending prompt to OpenAI GPT-4o-mini...", file=sys.stderr)
#    loop = asyncio.get_event_loop()
#
#    def sync_call():
#        response = client.chat.completions.create(
#            model=MODEL_NAME,
#            messages=[{"role": "user", "content": prompt}],
#            temperature=temperature,
#            max_tokens=1000,
#        )
#        return response.choices[0].message.content
#
#    try:
#        response_text = await asyncio.wait_for(loop.run_in_executor(None, sync_call), timeout=timeout)
#        print(" OpenAI responded.", file=sys.stderr)
#        return response_text.strip()
#    except asyncio.TimeoutError:
#        print(f" OpenAI timed out after {timeout} seconds.", file=sys.stderr)
#        return "other"
#    except Exception as e:
#        print(f" OpenAI failed: {e}", file=sys.stderr)
#        return "other"
async def ask_openai(prompt: str, timeout: int = 60, temperature: float = 0.0) -> str:
    """Send prompt to either OpenAI or Anthropic based on USE_ANTHROPIC_CLAUDE flag."""
    
    if llm_provider == "anthropic":
        print(f"[LLM] Using Anthropic Claude Sonnet 3.7 model", file=sys.stderr)
        try:
            loop = asyncio.get_event_loop()
            
            def sync_anthropic_call():
                response = anthropic_client.messages.create(
                    model="claude-3-7-sonnet-20250219",
                    max_tokens=1000,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
            
            response_text = await asyncio.wait_for(
                loop.run_in_executor(None, sync_anthropic_call), 
                timeout=timeout
            )
            print(f"[LLM] Anthropic Claude responded.", file=sys.stderr)
            return response_text.strip()
            
        except asyncio.TimeoutError:
            print(f"[LLM] Anthropic Claude timed out after {timeout} seconds.", file=sys.stderr)
            return "other"
        except Exception as e:
            print(f"[LLM] Anthropic Claude failed: {e}", file=sys.stderr)
            return "other"
    elif llm_provider == "openai":
        print(f"[LLM] Using OpenAI GPT-4o-mini model", file=sys.stderr)
        loop = asyncio.get_event_loop()

        def sync_call():
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1000,
            )
            return response.choices[0].message.content

        try:
            response_text = await asyncio.wait_for(loop.run_in_executor(None, sync_call), timeout=timeout)
            print(f"[LLM] OpenAI responded.", file=sys.stderr)
            return response_text.strip()
        except asyncio.TimeoutError:
            print(f"[LLM] OpenAI timed out after {timeout} seconds.", file=sys.stderr)
            return "other"
        except Exception as e:
            print(f"[LLM] OpenAI failed: {e}", file=sys.stderr)
            return "other"
    else:
        print(f"[LLM] Unknown provider: {llm_provider}", file=sys.stderr)
        return "other"
          
def extract_environment_id_from_testcases(test_cases: List[Dict[str, Any]]) -> Optional[int]:
    """
    Extract environment ID from test cases.
    Returns the first valid environment ID found, or None if not found.
    """
    print(f"DEBUG: Searching for environment ID in {len(test_cases)} test cases", file=sys.stderr)
    
    for tc in test_cases:
        tc_id = tc.get('tcSeqNo', tc.get('tcId', 'unknown'))
        print(f"DEBUG: Checking test case {tc_id} for environment ID", file=sys.stderr)
        
        # Check for environmentId directly in test case
        if "environmentId" in tc and tc["environmentId"] is not None:
            env_id = tc["environmentId"]
            if isinstance(env_id, int) and env_id > 0:
                print(f"DEBUG: Found environment ID: {env_id} in test case {tc_id}", file=sys.stderr)
                return env_id
        
        # Check nested in testData if it exists
        test_data = tc.get("testData", {})
        if isinstance(test_data, dict) and "environmentId" in test_data:
            env_id = test_data["environmentId"]
            if isinstance(env_id, int) and env_id > 0:
                print(f"DEBUG: Found environment ID: {env_id} in testData of test case {tc_id}", file=sys.stderr)
                return env_id
    
    print("DEBUG: No valid environment ID found in test cases", file=sys.stderr)
    return None
# ---------------------------------------------------------------------------
# WEBSOCKET UTILITIES
# ---------------------------------------------------------------------------
#async def send_request(ws, method: str, params: Dict[str, Any] = {}) -> Any:
#    """Sends a JSON-RPC request and waits for the response with better error handling"""
#    request_id = str(uuid.uuid4())
#    payload = {
#        "jsonrpc": "2.0",
#        "id": request_id,
#        "method": method,
#        "params": params,
#    }
#    
#    print(f" Sending {method} request (id: {request_id})", file=sys.stderr)
#    
#    try:
#        await ws.send(json.dumps(payload))
#        
#        # Wait for response with timeout
#        timeout = 30  # 30 seconds timeout
#        start_time = asyncio.get_event_loop().time()
#        
#        while True:
#            if asyncio.get_event_loop().time() - start_time > timeout:
#                print(f"Timeout waiting for {method} response", file=sys.stderr)
#                return None
#                
#            try:
#                response_raw = await asyncio.wait_for(ws.recv(), timeout=5)
#                response = json.loads(response_raw)
#                
#                print(f" Received response for {request_id}: {type(response.get('result', 'no result'))}", file=sys.stderr)
#                
#                if response.get("id") == request_id:
#                    if "error" in response:
#                        error = response["error"]
#                        print(f" Server error for {method}: {error.get('message', 'Unknown error')}", file=sys.stderr)
#                        return None
#                    return response.get("result")
#                else:
#                    print(f" Got response for different request: {response.get('id')}", file=sys.stderr)
#                    continue
#                    
#            except asyncio.TimeoutError:
#                continue  # Keep waiting
#            except json.JSONDecodeError as e:
#                print(f"Failed to parse server response: {e}", file=sys.stderr)
#                continue
#                
#    except Exception as e:
#        print(f"Error sending request for {method}: {e}", file=sys.stderr)
#        return None
async def authenticate_with_server(ws) -> bool:
    """Perform initial authentication with server using static credentials"""
    global CURRENT_JWT_TOKEN
    
    if not AGENT_ID or not AGENT_SECRET_KEY:
        print("  Missing agent credentials in environment variables", file=sys.stderr)
        print("  Please set AGENT_ID and AGENT_SECRET_KEY in .env file", file=sys.stderr)
        return False
    
    print(f" Authenticating agent: {AGENT_ID}", file=sys.stderr)
    print(f" Using secret key (length: {len(AGENT_SECRET_KEY)} chars)", file=sys.stderr)
    
    auth_request = {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "authenticate",
        "params": {
            "agentId": AGENT_ID,
            "secretKey": AGENT_SECRET_KEY
        }
    }
    
    try:
        await ws.send(json.dumps(auth_request))
        
        # Wait for authentication response
        timeout = 10  # 10 seconds timeout for auth
        start_time = asyncio.get_event_loop().time()
        
        while True:
            if asyncio.get_event_loop().time() - start_time > timeout:
                print(f"  Authentication timeout after {timeout} seconds", file=sys.stderr)
                return False
                
            try:
                response_raw = await asyncio.wait_for(ws.recv(), timeout=5)
                response = json.loads(response_raw)
                
                if response.get("id") == auth_request["id"]:
                    if "error" in response:
                        error = response["error"]
                        print(f"  Authentication failed: {error.get('message', 'Unknown error')}", file=sys.stderr)
                        return False
                    elif "result" in response:
                        result = response["result"]
                        if result.get("success"):
                            CURRENT_JWT_TOKEN = result.get("jwtToken", "")
                            print(f"[JWT] Authentication successful!", file=sys.stderr)
                            print(f"[JWT] JWT token received (length: {len(CURRENT_JWT_TOKEN)} chars)", file=sys.stderr)
                            print(f"[JWT] JWT token received: {CURRENT_JWT_TOKEN}", file=sys.stderr)
                    
                            return True
                        else:
                            print(f"  Authentication failed: {result}", file=sys.stderr)
                            return False
                    break
                else:
                    continue
                    
            except asyncio.TimeoutError:
                continue
            except json.JSONDecodeError as e:
                print(f"  Failed to parse authentication response: {e}", file=sys.stderr)
                return False
                
    except Exception as e:
        print(f"  Authentication error: {e}", file=sys.stderr)
        return False
    
    return False

async def send_request(ws, method: str, params: Dict[str, Any] = {}) -> Any:
    """Sends a JSON-RPC request and waits for the response with better error handling"""
    global CURRENT_JWT_TOKEN
    request_id = str(uuid.uuid4())
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params,
    }

    if method != "authenticate" and CURRENT_JWT_TOKEN:
        payload["jwtToken"] = CURRENT_JWT_TOKEN # JWT token added here

    
    print(f" Sending {method} request (id: {request_id})", file=sys.stderr)
    if CURRENT_JWT_TOKEN and method != "authenticate":
        print(f" Including JWT token (first 20 chars): {CURRENT_JWT_TOKEN[:20]}...", file=sys.stderr)
    
    try:
        await ws.send(json.dumps(payload))
        
        # Wait for response with timeout
        timeout = 30  # 30 seconds timeout
        start_time = asyncio.get_event_loop().time()
        
        while True:
            if asyncio.get_event_loop().time() - start_time > timeout:
                print(f"Timeout waiting for {method} response", file=sys.stderr)
                return None
                
            try:
                response_raw = await asyncio.wait_for(ws.recv(), timeout=5)
                if await handle_websocket_message(ws, response_raw):
                    continue

                response = json.loads(response_raw)
                
                print(f" Received response for {request_id}: {type(response.get('result', 'no result'))}", file=sys.stderr)
                
                if response.get("id") == request_id:
                    if "error" in response:
                        error = response["error"]
                        error_message = error.get('message', 'Unknown error')
                        error_code = error.get('code', -32000)
                        error_data = error.get('data', {})
                        
                        if error_code == 4004 and 'newToken' in error_data:
                           new_token = error_data.get('newToken')
                           expires_in = error_data.get('expiresIn')
                           
                           print(f"[JWT] Token expired - updating with new token from server", file=sys.stderr)
                           print(f"[JWT] New token expires in: {expires_in}", file=sys.stderr)
                           
                           CURRENT_JWT_TOKEN = new_token
                           
                           print(f"[JWT] Retrying {method} request with refreshed token", file=sys.stderr)
                           retry_payload = payload.copy()
                           retry_payload["jwtToken"] = CURRENT_JWT_TOKEN
                           
                           await ws.send(json.dumps(retry_payload))
                           continue 
                        # Enhanced debug logging for database errors
                        print(f" SERVER ERROR for {method}: {error_message}", file=sys.stderr)
                        print(f" Error Code: {error_code}", file=sys.stderr)
                        
                        if error_data:
                            if 'environmentId' in error_data:
                                print(f" Environment ID: {error_data['environmentId']}", file=sys.stderr)
                            if 'originalError' in error_data:
                                print(f" Original Error: {error_data['originalError']}", file=sys.stderr)
                            if 'errorCode' in error_data:
                                print(f" Database Error Code: {error_data['errorCode']}", file=sys.stderr)
                        
                        # Special handling for connection timeout errors
                        if error_code in [-32001, -32002]:  # Custom connection error codes
                            print(f" DATABASE CONNECTION ISSUE DETECTED:", file=sys.stderr)
                            print(f" - Method: {method}", file=sys.stderr)
                            print(f" - Environment ID: {params.get('environmentId', 'Not specified')}", file=sys.stderr)
                            print(f" - Error Type: {'Connection Timeout' if error_code == -32001 else 'Connection Refused'}", file=sys.stderr)
                        
                        return None
                    return response.get("result")
                else:
                    print(f" Got response for different request: {response.get('id')}", file=sys.stderr)
                    continue
                    
            except asyncio.TimeoutError:
                continue  # Keep waiting
            except json.JSONDecodeError as e:
                print(f"Failed to parse server response: {e}", file=sys.stderr)
                continue
                
    except Exception as e:
        print(f"Error sending request for {method}: {e}", file=sys.stderr)
        return None
    
# ---------------------------------------------------------------------------
# DATABASE OPERATIONS
# ---------------------------------------------------------------------------
async def get_db_schema(ws):
    """
    Fetch database schema from MCP server with environment ID context.
    NOTE: This should be called AFTER test cases are loaded to get correct environment ID.
    """
    global DB_SCHEMA, CURRENT_ENVIRONMENT_ID
    
    # Extract environment ID from loaded test cases
    if TEST_CASES:
        CURRENT_ENVIRONMENT_ID = extract_environment_id_from_testcases(TEST_CASES)
        if CURRENT_ENVIRONMENT_ID is None:
            print("ERROR: No environment ID found in test cases. Cannot proceed without environment ID.", file=sys.stderr)
            raise ValueError("Environment ID is required but not found in test cases")
        
        print(f"DEBUG: Using environment ID: {CURRENT_ENVIRONMENT_ID} for database schema", file=sys.stderr)
    else:
        print("ERROR: No test cases loaded. Cannot determine environment ID.", file=sys.stderr)
        raise ValueError("Test cases must be loaded before getting database schema")
    
    print(f"DEBUG: Sending environment ID {CURRENT_ENVIRONMENT_ID} to server", file=sys.stderr)
    
    # Send request with environment ID
    result = await send_request(ws, "DBSchema", {
        "ctx": {},
        "environmentId": CURRENT_ENVIRONMENT_ID
    })
    
    print(f"DEBUG: Server responded for environment ID {CURRENT_ENVIRONMENT_ID}", file=sys.stderr)
    print_raw_response("DBSchema", result, "call_dbschema")
    
    try:
        DB_SCHEMA = json.loads(result) if isinstance(result, str) else result or {}
    except Exception:
        DB_SCHEMA = {}
    print(f"DEBUG: DBSchema tables loaded for environment {CURRENT_ENVIRONMENT_ID}: {len(DB_SCHEMA)}", file = sys.stderr)

def debug_schema_structure():
    """Debug function to understand DB_SCHEMA structure"""
    print("\n DB_SCHEMA Debug Information:", file=sys.stderr)
    print(f" Schema type: {type(DB_SCHEMA)}", file=sys.stderr)
    
    if not isinstance(DB_SCHEMA, dict):
        print(" Schema is not a dictionary!", file=sys.stderr)
        return
        
    print(f" Schema keys (tables): {list(DB_SCHEMA.keys())}", file=sys.stderr)
    
    if not DB_SCHEMA:
        print(" Schema is empty! Check your MCP server connection.", file=sys.stderr)
        return
    
    # Show first table structure
    first_table_name = list(DB_SCHEMA.keys())[0]
    first_table = DB_SCHEMA[first_table_name]
    print(f" First table '{first_table_name}' structure:", file=sys.stderr)
    print(f" Table type: {type(first_table)}", file=sys.stderr)
    
    if isinstance(first_table, list) and first_table:
        print(f" First column structure: {first_table[0]}", file=sys.stderr)
        print(f" Column keys: {list(first_table[0].keys()) if isinstance(first_table[0], dict) else 'Not a dict'}", file=sys.stderr)
    
    # Show actual structure sample (truncated)
    try:
        sample = json.dumps(first_table, indent=2)[:500]
        print(f" First table sample:\n{sample}...", file=sys.stderr)
    except Exception as e:
        print(f" Could not serialize table sample: {e}", file=sys.stderr)

def extract_schema_columns(table_name: str) -> List[str]:
    """Extract column names from schema based on your MCP server format"""
    if table_name not in DB_SCHEMA:
        return []
    
    table_info = DB_SCHEMA[table_name]
    columns = []
    
    # Your MCP server format: [{"Field": "col1", "Type": "varchar", ...}, ...]
    if isinstance(table_info, list):
        for col_info in table_info:
            if isinstance(col_info, dict):
                # Your server uses "Field" key for column names
                if "Field" in col_info:
                    columns.append(col_info["Field"])
                elif "name" in col_info:  # Fallback
                    columns.append(col_info["name"])
    
    # Handle other possible formats
    elif isinstance(table_info, dict):
        if "columns" in table_info:
            cols = table_info["columns"]
            if isinstance(cols, list):
                for col in cols:
                    if isinstance(col, dict):
                        if "Field" in col:
                            columns.append(col["Field"])
                        elif "name" in col:
                            columns.append(col["name"])
                    else:
                        columns.append(str(col))
        else:
            # Direct key-value format
            columns = list(table_info.keys())
    
    #print(f" Extracted columns for {table_name}: {columns}", file=sys.stderr)
    return columns

async def execute_sql_query(ws, sql: str) -> List[Dict[str, Any]]:
    """
    Execute SQL query and return results with improved error handling.
    NOW HANDLES TIERED FALLBACK QUERIES for relational consistency.
    """
    # Handle special tiered fallback queries
    if sql.startswith("__CONTEXT_WITH_TIERED_FALLBACK__"):
        parts = sql.split("::")
        if len(parts) >= 2:
            query_chain = parts[1].split(" | ")
            
            print(f" Attempting tiered fallback with {len(query_chain)} queries", file=sys.stderr)
            
            # Try each query in order until one returns results
            for i, query in enumerate(query_chain):
                print(f" Attempting tier {i+1}: {query}", file=sys.stderr)
                results = await _execute_single_sql(ws, query.strip())
                
                if results:
                    print(f" Tier {i+1} successful: {len(results)} rows", file=sys.stderr)
                    return results
                else:
                    print(f" Tier {i+1} returned zero rows, trying next tier", file=sys.stderr)
            
            print(f" All tiers failed, returning empty results", file=sys.stderr)
            return []
        else:
            print(f" Invalid tiered fallback format: {sql}", file=sys.stderr)
            return []
    
    # Handle special context-with-fallback queries (existing functionality)
    if sql.startswith("__CONTEXT_WITH_FALLBACK__"):
        parts = sql.split("::")
        if len(parts) >= 3:
            context_sql = parts[1]
            fallback_sql = parts[2]
            
            print(f" Attempting context-based query first: {context_sql}", file=sys.stderr)
            
            # Try context query first
            context_results = await _execute_single_sql(ws, context_sql)
            
            if context_results:
                print(f" Context query successful: {len(context_results)} rows", file=sys.stderr)
                return context_results
            else:
                print(f" Context query returned zero rows, falling back to: {fallback_sql}", file=sys.stderr)
                fallback_results = await _execute_single_sql(ws, fallback_sql)
                print(f" Fallback query returned: {len(fallback_results)} rows", file=sys.stderr)
                return fallback_results
        else:
            print(f" Invalid context-with-fallback format: {sql}", file=sys.stderr)
            return []
    
    # Handle regular SQL queries
    return await _execute_single_sql(ws, sql)
    
async def _execute_single_sql(ws, sql: str) -> List[Dict[str, Any]]:
    """
    Execute a single SQL query and return results - internal helper function.
    """
    # Clean the SQL string first
    cleaned_sql = sql.strip()
    
    # Debug: Show the raw SQL
    print(f" Executing SQL: {cleaned_sql}", file=sys.stderr)
    print(f" SQL SENT TO DB: '{cleaned_sql}'", file=sys.stderr)
    
    if not cleaned_sql:
        print(" Empty SQL query provided", file=sys.stderr)
        return []
    
    try:
        result = await send_request(ws, "execute_query", {
            "ctx": {},
            "sql": cleaned_sql,
            "sql_parameters": []
        })

        print(f" Raw result type: {type(result)}", file=sys.stderr)

        if result is None:
            print(" Server returned None", file=sys.stderr)
            return []

        # Handle the response based on its type
        rows = []
        if isinstance(result, list):
            # Server returned data directly as list
            rows = result
            print(f" Got direct list with {len(rows)} rows", file=sys.stderr)
        elif isinstance(result, str):
            # Server returned JSON string (old behavior)
            try:
                rows = json.loads(result)
                print(f" Parsed JSON string to get {len(rows) if isinstance(rows, list) else 0} rows", file=sys.stderr)
            except json.JSONDecodeError as e:
                print(f" Failed to parse JSON response: {e}", file=sys.stderr)
                print(f" Raw response: {result[:200]}...", file=sys.stderr)
                return []
        else:
            print(f" Unexpected result type: {type(result)}", file=sys.stderr)
            return []

        # Ensure we have a list of dictionaries
        if not isinstance(rows, list):
            print(f" Expected list, got {type(rows)}", file=sys.stderr)
            return []

        # Validate the rows
        valid_rows = []
        for i, row in enumerate(rows):
            if isinstance(row, dict):
                valid_rows.append(row)
            else:
                print(f" Row {i} is not a dictionary: {type(row)}", file=sys.stderr)

        print(f" SQL returned {len(valid_rows)} valid rows", file=sys.stderr)
        if valid_rows:
            sample_keys = list(valid_rows[0].keys())[:5]  # Show first 5 keys
            print(f" Sample row keys: {sample_keys}", file=sys.stderr)
        
        return valid_rows

    except Exception as e:
        print(f" Error executing SQL: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return []
    
# ---------------------------------------------------------------------------
# NEW RELATIONAL CONSISTENCY FUNCTIONS
# ---------------------------------------------------------------------------

def build_relationship_inference_layer(db_schema: Dict[str, Any]) -> Dict[str, Dict[str, List[str]]]:
    """
    Build Relationship Inference Layer (RIL) dynamically from schema.
    Detects shared keys between tables using pattern matching.
    Returns: {table_name: {key_column: [linked_table_names]}}
    """
    if not db_schema:
        print(" No schema available for RIL building", file=sys.stderr)
        return {}
    
    # Extract all table columns first
    table_columns = {}
    for table_name, table_info in db_schema.items():
        columns = extract_schema_columns(table_name)
        table_columns[table_name] = columns
    
    #print(f" Building RIL from {len(table_columns)} tables", file=sys.stderr)
    
    relationship_graph = {}
    
    for source_table, source_columns in table_columns.items():
        relationship_graph[source_table] = {}
        
        # Find key-like columns in source table
        key_columns = []
        for column in source_columns:
            # Pattern 1: ends with _id or is just 'id'
            if column.lower().endswith('_id') or column.lower() == 'id':
                key_columns.append(column)
        
        # For each key column, find other tables that have the same column
        for key_column in key_columns:
            linked_tables = []
            
            for target_table, target_columns in table_columns.items():
                if target_table != source_table:
                    # Check for exact name match (case insensitive)
                    for target_column in target_columns:
                        if key_column.lower() == target_column.lower():
                            linked_tables.append(target_table)
                            break
            
            if linked_tables:
                relationship_graph[source_table][key_column] = linked_tables
                #print(f" RIL: {source_table}.{key_column} links to tables: {linked_tables}", file=sys.stderr)
    
    # Remove empty relationships
    relationship_graph = {table: relationships for table, relationships in relationship_graph.items() if relationships}
    
    print(f" RIL built successfully: {len(relationship_graph)} tables with relationships", file=sys.stderr)
    return relationship_graph

def extract_key_columns(row_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract key-like columns from a database row.
    Returns dictionary of {column_name: value} for potential linking keys.
    """
    key_data = {}
    
    for column, value in row_data.items():
        # Skip internal metadata
        if column == "__table__":
            continue
            
        # Pattern matching for key columns
        column_lower = column.lower()
        if (column_lower.endswith('_id') or 
            column_lower == 'id' or
            column_lower.endswith('_key') or
            column_lower.endswith('_code')):
            key_data[column] = value
    
    return key_data

def update_memory_key_index(row_data: Dict[str, Any], memory_key_index: Dict[str, List[Any]]):
    """
    Update the memory key index with keys from a newly stored row.
    Modifies memory_key_index in place.
    """
    key_data = extract_key_columns(row_data)
    
    for key_column, key_value in key_data.items():
        if key_column not in memory_key_index:
            memory_key_index[key_column] = []
        
        # Only add if not already present
        if key_value not in memory_key_index[key_column]:
            memory_key_index[key_column].append(key_value)
            print(f" Memory key index updated: {key_column} = {key_value}", file=sys.stderr)

def find_relational_constraints(
    target_table: str, 
    relationship_graph: Dict[str, Dict[str, List[str]]], 
    memory_key_index: Dict[str, List[Any]]
) -> Dict[str, Any]:
    """
    Find relational constraints for linking target table to memory entities.
    Returns dictionary of {column_name: value} for WHERE conditions.
    """
    relational_constraints = {}
    
    if not relationship_graph or not memory_key_index:
        return relational_constraints
    
    print(f" Finding relational constraints for table: {target_table}", file=sys.stderr)
    print(f" Available memory keys: {list(memory_key_index.keys())}", file=sys.stderr)
    
    # Look for any table that can link to the target table
    for source_table, relationships in relationship_graph.items():
        for link_key, linked_tables in relationships.items():
            # If target table is in the linked tables and we have this key in memory
            if target_table in linked_tables and link_key in memory_key_index:
                # Use the most recent value for this key
                key_values = memory_key_index[link_key]
                if key_values:
                    # For now, use the last (most recent) value
                    relational_constraints[link_key] = key_values[-1]
                    print(f" Found relational link: {target_table}.{link_key} = {key_values[-1]} (via {source_table})", file=sys.stderr)
                    # Only use one link for now to keep it simple
                    break
        
        if relational_constraints:
            break
    
    return relational_constraints

def build_tiered_query_strategy(
    base_sql: str,
    literal_conditions: Dict[str, Any],
    relational_constraints: Dict[str, Any],
    table_name: str
) -> str:
    """
    Build tiered query strategy with graceful fallbacks.
    Returns SQL string with embedded fallback logic.
    """
    # Helper function to build WHERE clause
    def build_where_clause(conditions: Dict[str, Any]) -> str:
        if not conditions:
            return ""
        
        where_parts = []
        for field, value in conditions.items():
            if value is not None:
                if isinstance(value, str):
                    if value.isdigit():
                        clause = f"{field} = {value}"
                    else:
                        escaped_value = value.replace("'", "''")
                        clause = f"{field} = '{escaped_value}'"
                elif isinstance(value, (int, float)):
                    clause = f"{field} = {value}"
                else:
                    clause = f"{field} = '{value}'"
                where_parts.append(clause)
        
        return " WHERE " + " AND ".join(where_parts) if where_parts else ""
    
    # Build different query tiers
    queries = []
    
    # Tier 1: Literals + Relational (most specific)
    if literal_conditions and relational_constraints:
        combined_conditions = {**literal_conditions, **relational_constraints}
        tier1_sql = base_sql + build_where_clause(combined_conditions) + " LIMIT 1"
        queries.append(tier1_sql)
        print(f" Tier 1 (literal + relational): {tier1_sql}", file=sys.stderr)
    
    # Tier 2: Literals only (current behavior)
    if literal_conditions:
        tier2_sql = base_sql + build_where_clause(literal_conditions) + " LIMIT 1"
        queries.append(tier2_sql)
        print(f" Tier 2 (literal only): {tier2_sql}", file=sys.stderr)
    
    # Tier 3: Relational only (new consistency feature)
    if relational_constraints:
        tier3_sql = base_sql + build_where_clause(relational_constraints) + " ORDER BY RAND() LIMIT 1"
        queries.append(tier3_sql)
        print(f" Tier 3 (relational only): {tier3_sql}", file=sys.stderr)
    
    # Tier 4: Random fallback (unchanged behavior)
    tier4_sql = base_sql + " ORDER BY RAND() LIMIT 1"
    queries.append(tier4_sql)
    print(f" Tier 4 (random fallback): {tier4_sql}", file=sys.stderr)
    
    # Return as context-with-fallback format with multiple tiers
    if len(queries) > 1:
        return f"__CONTEXT_WITH_TIERED_FALLBACK__::{' | '.join(queries)}"
    else:
        return queries[0]
    
# ---------------------------------------------------------------------------
# NEW HELPER FUNCTIONS 
# ---------------------------------------------------------------------------

def transform_value_based_on_action(value: str, action: str) -> str:
    """
    Dynamically transform a database-extracted value based on keywords in the action text:
    - 'partial'   → return a random substring of length len(value)//2
    - 'uppercase' → return value.upper()
    - 'lowercase' → return value.lower()
    Otherwise, return the original value.
    """
    if not isinstance(value, str):
        return value

    action_lower = action.lower()

    # Partial: half-length random slice
    if "partial" in action_lower:
        half = max(1, len(value) // 2)
        if len(value) > half:
            start = random.randint(0, len(value) - half)
            return value[start : start + half]
        return value

    # Case transformations
    if "uppercase" in action_lower:
        return value.upper()
    if "lowercase" in action_lower:
        return value.lower()

    return value

def flatten_memory_data(memory_obj) -> Dict[str, Any]:
    """Flatten memory data into a dictionary with table-scoped keys"""
    flattened = {}
    for idx, entity in enumerate(memory_obj.get_all()):
        if isinstance(entity, dict):
            table_name = entity.get("__table__", "unknown").lower()  # fallback to 'unknown' if missing
            for key, value in entity.items():
                if key == "__table__":
                    continue  # skip internal metadata
                safe_key = f"row_{idx + 1}.{table_name}.{key}"
                flattened[safe_key] = value
    return flattened



def find_table_for_field(field_name: str) -> Optional[str]:
    """Find which table contains a specific field - updated for your MCP server format"""
    for table_name, table_info in DB_SCHEMA.items():
        if isinstance(table_info, list):
            # Your MCP server format: [{"Field": "col1", ...}, ...]
            for col_info in table_info:
                if isinstance(col_info, dict) and col_info.get("Field") == field_name:
                    return table_name
        elif isinstance(table_info, dict) and "columns" in table_info:
            # Standard format fallback
            for col in table_info["columns"]:
                if isinstance(col, dict) and col.get("name") == field_name:
                    return table_name
    return None


async def analyze_table_relationships(source_table: str, target_table: str) -> Dict[str, Any]:
    """Find common fields and possible join paths between tables - updated for your MCP server format"""
    if not DB_SCHEMA:
        return {}
    
    source_columns = set()
    target_columns = set()
    
    # Extract columns from source table using your MCP server format
    if source_table in DB_SCHEMA:
        source_columns = set(extract_schema_columns(source_table))
    
    # Extract columns from target table using your MCP server format
    if target_table in DB_SCHEMA:
        target_columns = set(extract_schema_columns(target_table))
    
    # Find common fields
    common_fields = source_columns.intersection(target_columns)
    
    # Find potential linking fields (foreign keys, IDs, etc.)
    linking_patterns = ["id", "name", "email", "phone", "code", "number"]
    potential_links = []
    
    for field in common_fields:
        if any(pattern in field.lower() for pattern in linking_patterns):
            potential_links.append(field)
    
    # Sort by priority (id fields first, then name fields, etc.)
    potential_links.sort(key=lambda x: (
        0 if "id" in x.lower() else
        1 if "name" in x.lower() else
        2 if "email" in x.lower() else
        3
    ))
    
    return {
        "common_fields": list(common_fields),
        "potential_links": potential_links,
        "source_columns": list(source_columns),
        "target_columns": list(target_columns)
    }

def debug_memory_state(memory, label: str):
    """Debug helper to show current memory state"""
    entities = memory.get_all()
    print(f"\n {label} Memory State:", file=sys.stderr)
    print(f"Total entities: {len(entities)}", file=sys.stderr)
    
    for idx, entity in enumerate(entities, 1):
        if isinstance(entity, dict):
            print(f"  [{idx}] Fields: {list(entity.keys())}", file=sys.stderr)
            # Show a few sample values
            sample_items = list(entity.items())[:3]
            for key, value in sample_items:
                print(f"      {key}: {value}", file=sys.stderr)
            if len(entity) > 3:
                print(f"      ... and {len(entity) - 3} more fields", file=sys.stderr)
        else:
            print(f"  [{idx}] {entity}", file=sys.stderr)

async def resolve_field_with_memory_intelligence(ws, target_field: str, target_table: str, memory_data: Dict[str, Any]) -> Optional[Any]:
    """
    Intelligently resolve a field using memory data and schema relationships
    """
    print(f" Resolving field: {target_field} using memory intelligence", file=sys.stderr)
    print(f" Available memory data: {list(memory_data.keys())}", file=sys.stderr)
    
    # First, try direct lookup in memory
    if target_field in memory_data:
        result = memory_data[target_field]
        print(f" Direct memory lookup succeeded for {target_field}: {result}", file=sys.stderr)
        return result
    
    for memory_field, memory_value in memory_data.items():
        if (target_field.lower() in memory_field.lower() or 
            memory_field.lower() in target_field.lower() or
            target_field.replace('_', '').lower() == memory_field.replace('_', '').lower()):
            print(f" Fuzzy match found: {target_field} ≈ {memory_field} = {memory_value}", file=sys.stderr)
            return memory_value

    # If direct lookup fails, analyze relationships
    print(f" Direct lookup failed, analyzing schema relationships...", file=sys.stderr)
    
    # Try to find the target table and analyze relationships with known tables
    for known_field, known_value in memory_data.items():
        # Find which table this known field belongs to
        source_table = find_table_for_field(known_field)
        if not source_table:
            continue
            
        print(f" Analyzing relationship: {source_table} → {target_table} using {known_field}", file=sys.stderr)
        
        # Analyze relationship between source and target tables
        relationships = await analyze_table_relationships(source_table, target_table)
        
        if relationships["potential_links"]:
            print(f" Found potential linking fields: {relationships['potential_links']}", file=sys.stderr)
            
            # Try using common fields as bridges
            for link_field in relationships["potential_links"]:
                try:
                    # Get the linking value from source table
                    bridge_sql = f"SELECT {link_field} FROM {source_table} WHERE {known_field} = '{known_value}'"
                    print(f" Bridge lookup SQL: {bridge_sql}", file=sys.stderr)
                    bridge_rows = await execute_sql_query(ws, bridge_sql)
                    
                    if bridge_rows and link_field in bridge_rows[0]:
                        bridge_value = bridge_rows[0][link_field]
                        print(f" Bridge value found: {link_field} = {bridge_value}", file=sys.stderr)
                        
                        # Now use bridge value to get target field from target table
                        target_sql = f"SELECT {target_field} FROM {target_table} WHERE {link_field} = '{bridge_value}'"
                        print(f" Target lookup SQL: {target_sql}", file=sys.stderr)
                        target_rows = await execute_sql_query(ws, target_sql)
                        
                        if target_rows and target_field in target_rows[0]:
                            result = target_rows[0][target_field]
                            print(f" Memory intelligence resolved {target_field} = {result}", file=sys.stderr)
                            return result
                        else:
                            print(f" Target field {target_field} not found in result", file=sys.stderr)
                            
                except Exception as e:
                    print(f" Bridge lookup failed for {link_field}: {e}", file=sys.stderr)
                    continue
    
    print(f" Memory intelligence failed to resolve {target_field}", file=sys.stderr)
    return None

async def enhance_where_conditions_with_memory(where_conditions: Dict[str, Any], memory_data: Dict[str, Any], target_table: str, ws) -> Dict[str, Any]:
    """
    Enhanced WHERE conditions with table-scoped memory filtering and better value handling.
    Only uses memory from the same table to prevent cross-entity contamination.
    """
    enhanced_conditions = {}
    if not where_conditions:
        return enhanced_conditions

    print(f" Enhancing WHERE conditions for table: {target_table}", file=sys.stderr)
    print(f" Memory data (all tables): {memory_data}", file=sys.stderr)
    print(f" Original WHERE conditions: {where_conditions}", file=sys.stderr)

    #  Filter memory keys to only include those from the correct table
    filtered_memory = {}
    for key, value in memory_data.items():
        if f".{target_table.lower()}." in key.lower():
            field_name = key.split(".")[-1]  # extract just the field name
            filtered_memory[field_name] = value

    print(f" Filtered memory for table [{target_table}]: {filtered_memory}", file=sys.stderr)

    for field, value in where_conditions.items():
        #  Skip placeholder values
        if isinstance(value, str) and value.strip().lower() == "literal_value":
            print(f" Ignoring placeholder literal value for field: {field}", file=sys.stderr)
            continue

        #  Fix indirect field reference pattern
        if field.strip().lower() == "field" and isinstance(value, str):
            actual_field_name = value
            if actual_field_name in filtered_memory:
                actual_value = filtered_memory[actual_field_name]
                enhanced_conditions[actual_field_name] = actual_value
                print(f" Fixed 'field': '{value}' → '{actual_field_name}': '{actual_value}'", file=sys.stderr)
                continue
            else:
                print(f" Could not find '{actual_field_name}' in memory for table {target_table}", file=sys.stderr)
                continue

        #  Memory directive (e.g., "use value from memory")
        if isinstance(value, str) and ("use value from memory" in value.lower() or "use" in value.lower()):
            print(f" Resolving field: {field} with value: {value}", file=sys.stderr)

            resolved_value = await resolve_field_with_memory_intelligence(ws, field, target_table, memory_data)
            if resolved_value is not None:
                enhanced_conditions[field] = resolved_value
                print(f" Resolved {field} = {resolved_value} using memory intelligence", file=sys.stderr)
                continue

            # Fallback direct lookup — only use table-filtered memory
            if field in filtered_memory:
                enhanced_conditions[field] = filtered_memory[field]
                print(f" Resolved {field} = {filtered_memory[field]} from scoped memory", file=sys.stderr)
            else:
                print(f" Could not resolve {field} from memory for table {target_table}", file=sys.stderr)

        #  Use literal values from action text directly
        elif value is not None and isinstance(value, (str, int, float)) and value != "":
            enhanced_conditions[field] = value
            print(f" Using literal value from action: {field} = {value}", file=sys.stderr)

    print(f" Enhanced WHERE conditions: {enhanced_conditions}", file=sys.stderr)
    return enhanced_conditions



# ---------------------------------------------------------------------------
# TEST CASE OPERATIONS
# ---------------------------------------------------------------------------
#async def read_test_cases(ws=None):
#    """Load test cases directly from JSON passed via command-line argument."""
#    global TEST_CASES
#
#    # If a JSON string is passed as the first CLI argument, use that
#    if len(sys.argv) > 1:
#        try:
#            arg = sys.argv[1]
#            parsed = json.loads(arg)
#            print("Test case data: ", parsed)
#            # Wrap single object in a list if necessary
#            if isinstance(parsed, dict):
#                TEST_CASES = [parsed]
#            else:
#                TEST_CASES = parsed
#            print(f" Loaded {len(TEST_CASES)} test cases from argument.", file=sys.stderr)
#            return
#        except json.JSONDecodeError as e:
#            print(f"  Error parsing JSON from argument: {e}", file=sys.stderr)
#        except Exception as e:
#            print(f"  Unexpected error reading test cases from argument: {e}", file=sys.stderr)
#
    # If no valid argument, fallback to empty list
 #   TEST_CASES = []
 #   print("  No JSON argument provided or parsing failed.
 #  Loaded 0 test cases.", file=sys.stderr)
async def read_test_cases(ws=None):
    """Load test cases directly from JSON passed via command-line argument."""
    global TEST_CASES

    # If a JSON string is passed as the first CLI argument, use that
    if len(sys.argv) > 1:
        try:
            arg = sys.argv[1]
            parsed = json.loads(arg)
            # Wrap single object in a list if necessary
            if isinstance(parsed, dict):
                TEST_CASES = [parsed]
            else:
                TEST_CASES = parsed
            print(f"Loaded {len(TEST_CASES)} test cases from argument.", file=sys.stderr)
            return
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from argument: {e}", file=sys.stderr)
        except Exception as e:
            print(f"Unexpected error reading test cases from argument: {e}", file=sys.stderr)

    # If no valid argument, this is an error - we require test cases with environment ID
    print("ERROR: No test cases provided. Test cases with environment ID are required.", file=sys.stderr)
    raise ValueError("Test cases with environment ID must be provided as command line argument")

# ---------------------------------------------------------------------------
# IMPROVED SELECTIONAL PROCESSING
# ---------------------------------------------------------------------------
async def identify_selectional_steps(tc: Dict[str, Any], selectional_memory) -> List[Dict[str, Any]]:
    """
    UPDATED: Only force fields to creational if NO matches (exact OR LLM semantic) are found in the entire schema.
    Now uses LLM-powered semantic understanding instead of hardcoded patterns.
    """
    steps = tc.get("stepArray", [])
    if not steps:
        return []

    # Get initial field classification
    field_analysis = await analyze_fields_for_tc(tc)
    creational_fields = set(field_analysis.get("creational", []))
    selectional_fields = set(field_analysis.get("selectional", []))

    print(f" Field classification - Selectional: {selectional_fields}, Creational: {creational_fields}", file=sys.stderr)

    selectional_steps = []
    bracketed_pattern = r'<<([^>]+)>>'

    # Track fields that have NO database matches at all (for forced creational)
    completely_unmatched_fields = set()

    for step_index, step in enumerate(steps):
        step_number = step_index + 1
        action = step.get("tcStep", "")

        # Extract all <<field>> patterns from this step
        bracketed_fields = re.findall(bracketed_pattern, action)
        if not bracketed_fields:
            print(f" Skipping step {step_number} - no <<field>> patterns found: {action}", file=sys.stderr)
            continue

        # Only keep fields originally classified as selectional
        selectional_fields_in_step = [field for field in bracketed_fields if field in selectional_fields]
        if not selectional_fields_in_step:
            # Step has bracketed fields but they're all creational
            print(f" Skipping step {step_number} - only contains creational fields: {bracketed_fields}", file=sys.stderr)
            continue

        # ENHANCED CHECK: Verify these selectional fields can actually be found in schema
        # Get all schema fields for comprehensive checking
        all_schema_fields = {}
        for table_name, table_info in DB_SCHEMA.items():
            table_fields = extract_schema_columns(table_name)
            all_schema_fields[table_name] = table_fields

        # Check each selectional field for ANY possible match in schema
        fields_with_schema_matches = []
        for field in selectional_fields_in_step:
            has_any_match = False
            
            # Check for exact match across all tables
            for table_name, table_columns in all_schema_fields.items():
                for column in table_columns:
                    if normalize_field_name(field) == normalize_field_name(column):
                        has_any_match = True
                        print(f"   Field '{field}' has EXACT match with '{column}' in table '{table_name}'", file=sys.stderr)
                        break
                if has_any_match:
                    break
            
            # If no exact match, use LLM intelligent matching across all tables  
            if not has_any_match:
                for table_name, table_columns in all_schema_fields.items():
                    # Use LLM to check for semantic match in this table
                    llm_match = await llm_intelligent_field_selection(field, table_columns, action)
                    if llm_match and llm_match.confidence_score >= 70:
                        has_any_match = True
                        print(f"   Field '{field}' has LLM SEMANTIC match with '{llm_match.column_name}' in table '{table_name}' ({llm_match.confidence_score}%)", file=sys.stderr)
                        break
            
            if has_any_match:
                fields_with_schema_matches.append(field)
            else:
                completely_unmatched_fields.add(field)
                print(f"   Field '{field}' has NO matches in entire schema - will be forced to creational", file=sys.stderr)

        # Only proceed with analysis if there are fields that have schema matches
        if not fields_with_schema_matches:
            print(f" No selectional fields in step {step_number} have schema matches", file=sys.stderr)
            continue

        # Analyze this selectional step with only the fields that have schema matches
        memory_data = flatten_memory_data(selectional_memory)
        step_analysis = await analyze_selectional_step(step, tc, memory_data)

        if step_analysis:
            # Filter to only include fields that have schema matches
            step_analysis["stepNumber"] = step_number
            step_analysis["originalAction"] = step.get("tcStep", "")
            step_analysis["bracketedFields"] = fields_with_schema_matches  # Only fields with matches
            
            original_mapping = step_analysis.get("bracketed_fields_mapped", {})
            filtered_mapping = {k: v for k, v in original_mapping.items() if k in fields_with_schema_matches}
            step_analysis["bracketed_fields_mapped"] = filtered_mapping
            
            filtered_columns = [original_mapping.get(field, field) for field in fields_with_schema_matches]
            step_analysis["columns"] = filtered_columns

            selectional_steps.append(step_analysis)
            print(f" Added selectional step {step_number} with schema-matched fields: {fields_with_schema_matches}", file=sys.stderr)
        else:
            print(f" Could not analyze selectional step {step_number} even with schema-matched fields", file=sys.stderr)

    # Handle completely unmatched fields - force them to creational
    if completely_unmatched_fields:
        if "forced_creational_fields" not in tc:
            tc["forced_creational_fields"] = []
        
        for field in completely_unmatched_fields:
            if field not in tc["forced_creational_fields"]:
                tc["forced_creational_fields"].append(field)
        
        print(f" Fields with NO schema matches forced to creational: {completely_unmatched_fields}", file=sys.stderr)
        print(f" Total forced creational fields in this test case: {tc['forced_creational_fields']}", file=sys.stderr)

    print(f" Found {len(selectional_steps)} selectional steps with schema-matched fields", file=sys.stderr)
    return selectional_steps

async def validate_semantic_field_mapping_dynamic(action: str, selected_columns: List[str], available_columns: List[str], table_name: str, all_schema_fields: Dict[str, List[str]]) -> Dict[str, Any]:
    """
    Dynamically validate that selected columns semantically match the action intent using LLM analysis.
    Provides corrected columns if mismatches are detected.
    """
    print(f" Validating semantic mapping for action: {action}", file=sys.stderr)
    print(f" Selected columns: {selected_columns}", file=sys.stderr)
    print(f" Available columns: {available_columns}", file=sys.stderr)
    
    validation_prompt = f"""
You are a semantic field mapping validator. Your job is to determine if the selected database columns logically fulfill the intent of the given action.

Action: "{action}"
Selected Columns: {selected_columns}
Available Columns in Table '{table_name}': {available_columns}
All Available Fields Across All Tables: {all_schema_fields}

 **VALIDATION TASK:**

1. **ANALYZE ACTION INTENT**:
   - Understand what type of data the action is asking for
   - Identify the concept or purpose behind the requested field
   - Determine what kind of information would logically be needed

2. **EVALUATE SELECTED COLUMNS**:
   - For each selected column, determine if it logically serves the action's purpose
   - Consider if the column name and likely content would fulfill the action's intent
   - Look for semantic mismatches (e.g., selecting email when action asks for name/username)

3. **FIND BETTER ALTERNATIVES IF NEEDED**:
   - If selected columns don't match the action intent, find better alternatives
   - Look through ALL available columns to find ones that better serve the purpose
   - Consider functional equivalence and logical purpose matching

4. **PROVIDE REASONING**:
   - Explain why the selected columns do or don't match the action intent
   - If corrections are needed, explain why the alternative columns are better

Return your validation result in this exact JSON format:

{{
  "valid": true_or_false,
  "corrected_columns": ["list_of_best_columns_for_this_action"],
  "reasoning": "explanation of why columns were validated or corrected",
  "action_intent": "what type of data the action is asking for",
  "column_purpose_analysis": "analysis of what purpose each column serves"
}}

Be thorough in your analysis and prioritize logical purpose matching over any superficial name similarities.
"""

    try:
        result = await ask_openai(validation_prompt)
        
        if result.startswith("```"):
            result = "\n".join(line for line in result.splitlines() if not line.strip().startswith("```"))
        
        validation_result = json.loads(result)
        
        # Ensure corrected columns exist in the available columns
        corrected_columns = validation_result.get("corrected_columns", [])
        final_corrected_columns = [col for col in corrected_columns if col in available_columns]
        
        validation_output = {
            "valid": validation_result.get("valid", False),
            "corrected_columns": final_corrected_columns,
            "original_columns": selected_columns,
            "reasoning": validation_result.get("reasoning", ""),
            "action_intent": validation_result.get("action_intent", ""),
            "column_purpose_analysis": validation_result.get("column_purpose_analysis", "")
        }
        
        print(f" Dynamic semantic validation result: {validation_output}", file=sys.stderr)
        return validation_output
        
    except Exception as e:
        print(f" Error in dynamic semantic validation: {e}", file=sys.stderr)
        # Fallback to keeping original columns
        return {
            "valid": True,
            "corrected_columns": selected_columns,
            "original_columns": selected_columns,
            "reasoning": f"Validation failed due to error: {e}, keeping original columns",
            "action_intent": "Could not analyze",
            "column_purpose_analysis": "Validation error occurred"
        }


async def analyze_selectional_step(step: Dict[str, Any], tc: Dict[str, Any], memory_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    UPDATED: Analyze a selectional step with pre-validated semantic mappings.
    Now passes the semantic mapping information to the LLM to ensure correct field mapping.
    """
    import re

    # Ensure DB schema is loaded
    if not DB_SCHEMA:
        print(" Cannot analyze step - DB_SCHEMA is empty. Make sure to call get_db_schema() first!", file=sys.stderr)
        return None

    action = step.get("tcStep", "")
    expected_result = step.get("tcResult", "")

    # Extract <<field>> patterns
    bracketed_pattern = r'<<([^>]+)>>'
    bracketed_fields = re.findall(bracketed_pattern, action)

    if not bracketed_fields:
        print(f" No <<field>> patterns found in step: {action}", file=sys.stderr)
        return None

    print(f"---- FOUND BRACKETED FIELDS---- : {bracketed_fields} in step: {action}", file=sys.stderr)

    # Extract contextual steps for WHERE conditions
    all_steps = tc.get("stepArray", [])
    contextual_steps = []
    for other_step in all_steps:
        other_action = other_step.get("tcStep", "")
        if other_action != action and not re.search(bracketed_pattern, other_action):
            contextual_steps.append(other_action)
    print(f" Contextual steps for WHERE conditions: {contextual_steps}", file=sys.stderr)

    # Gather all schema fields
    all_schema_fields = {}
    for table_name, table_info in DB_SCHEMA.items():
        table_fields = extract_schema_columns(table_name)
        all_schema_fields[table_name] = table_fields

    # Pre-validate candidate tables and GET THE DETAILED MAPPINGS
    candidate_tables_with_mappings = await validate_tables_for_bracketed_fields_detailed(bracketed_fields, all_schema_fields, action)
    if not candidate_tables_with_mappings:
        print(f" No tables found that contain fields for bracketed patterns: {bracketed_fields}", file=sys.stderr)
        return None

    candidate_tables = list(candidate_tables_with_mappings.keys())
    print(f" Candidate tables that can satisfy bracketed fields: {candidate_tables}", file=sys.stderr)

    # Build the pre-validated mapping information for the LLM
    mapping_guidance = {}
    for table_name, mappings in candidate_tables_with_mappings.items():
        mapping_guidance[table_name] = {}
        for bracketed_field, (column_name, match_type) in mappings.items():
            mapping_guidance[table_name][bracketed_field] = {
                "column": column_name,
                "match_type": match_type,
                "explanation": f"{match_type} match to {column_name}"
            }

    #print(f" Pre-validated mapping guidance: {mapping_guidance}", file=sys.stderr)

    # ENHANCED prompt with pre-validated mapping guidance
    prompt = f"""
You are an expert test step analyzer that maps <<field>> patterns to database columns using PRE-VALIDATED MAPPINGS.

Context:
- Test Case Description: {tc.get("tcDescription", "")}
- Step Action: "{action}"
- Expected Result: "{expected_result}"
- Bracketed Fields Found: {bracketed_fields}
- Contextual Steps (for WHERE conditions): {contextual_steps}
- Memory Data Available: {list(memory_data.keys()) if memory_data else "None"}

Available Database Schema with ALL Fields:
{safe_dump(DB_SCHEMA)}

CRITICAL: You MUST choose from these validated candidate tables:
Candidate Tables: {candidate_tables}

**PRE-VALIDATED FIELD MAPPINGS (YOU MUST USE THESE):**
{safe_dump(mapping_guidance)}

 **MANDATORY FIELD MAPPING PROCESS:**

1. **USE PRE-VALIDATED MAPPINGS**:
   - The mappings above have been PRE-VALIDATED for exact and semantic matches
   - You MUST use these mappings - do not ignore them or create new mappings
   - For each bracketed field, use the corresponding column from the pre-validated mappings

2. **TABLE SELECTION PRIORITY**:
   - Select the table that has the most EXACT matches first
   - If no exact matches, select table with the most SEMANTIC matches
   - You MUST choose from candidate tables: {candidate_tables}

3. **FIELD MAPPING ENFORCEMENT**:
   - For each bracketed field in {bracketed_fields}, you MUST map it to its pre-validated column
   - Do NOT skip any bracketed fields - they all have valid mappings
   - Use the exact column names from the pre-validated mappings

4. **CONTEXTUAL WHERE CONDITION EXTRACTION**:
   - Analyze contextual steps: {contextual_steps}
   - Extract literal values and filter conditions
   - Build WHERE conditions using actual database column names

5. **MAPPING EXPLANATION REQUIREMENT**:
   - For each field mapping, use the explanation from the pre-validated mappings
   - Include whether it was EXACT or SEMANTIC matching

 **CRITICAL RULES:**
- You MUST map ALL bracketed fields: {bracketed_fields}
- You MUST use the pre-validated mappings provided above
- You MUST choose from candidate tables: {candidate_tables}
- Do NOT skip any bracketed fields - they all have valid database matches
- Use the pre-validated column names exactly as provided

 **SCHEMA FORMAT NOTE:**
The schema format is: {{"table_name": [{{"Field": "column_name", "Type": "data_type", ...}}, ...]}}
Always use the "Field" value as the actual column name.

Return your result in this exact JSON format:

{{
  "table": "actual_table_name_from_candidate_list",
  "columns": ["actual_schema_field_names_for_ALL_bracketed_fields"],
  "joins": [],
  "where_conditions": {{
    "actual_schema_field_name": "literal_value_from_contextual_steps_or_use_value_from_memory"
  }},
  "description": "What bracketed fields this step extracts and how contextual steps influence WHERE conditions",
  "depends_on_memory": true_or_false,
  "contextual_analysis": "explanation of how contextual steps were used for WHERE conditions",
  "bracketed_fields_mapped": {{"original_bracketed_field": "mapped_schema_column_from_pre_validation"}},
  "mapping_explanations": {{"original_bracketed_field": "explanation_from_pre_validation"}}
}}

IMPORTANT: You MUST include ALL bracketed fields in the output. Do not skip any fields.

If you cannot use the pre-validated mappings for some reason, return: null
"""

    result = await ask_openai(prompt)

    # Remove Markdown code fencing if LLM returns result inside ```
    if result.startswith("```"):
        result = "\n".join(line for line in result.splitlines() if not line.strip().startswith("```"))

    if result.strip().lower() == "null":
        print(" LLM returned null - could not use pre-validated mappings", file=sys.stderr)
        return None

    try:
        analysis = json.loads(result)
        if not isinstance(analysis, dict):
            print(f" Invalid analysis format: {type(analysis)}", file=sys.stderr)
            return None

        # Extract and validate results
        table_name = analysis.get("table")
        columns = analysis.get("columns", [])
        where_conditions = analysis.get("where_conditions", {})
        contextual_analysis = analysis.get("contextual_analysis", "")
        bracketed_mapping = analysis.get("bracketed_fields_mapped", {})
        mapping_explanations = analysis.get("mapping_explanations", {})

        print(f" Validating table: {table_name}, columns: {columns}", file=sys.stderr)
        print(f" WHERE conditions from context: {where_conditions}", file=sys.stderr)
        print(f" Contextual analysis: {contextual_analysis}", file=sys.stderr)
        print(f" Bracketed field mapping: {bracketed_mapping}", file=sys.stderr)
        print(f" Mapping explanations: {mapping_explanations}", file=sys.stderr)

        # Validate table selection
        if not table_name or table_name not in DB_SCHEMA:
            print(f" Table '{table_name}' not found in schema. Available: {list(DB_SCHEMA.keys())}", file=sys.stderr)
            return None
        if table_name not in candidate_tables:
            print(f" Table '{table_name}' not in validated candidate list: {candidate_tables}", file=sys.stderr)
            return None

        # Validate columns exist in selected table
        schema_columns = extract_schema_columns(table_name)
        if not schema_columns:
            print(f" No columns found for table {table_name}", file=sys.stderr)
            return None

        valid_columns = [col for col in columns if col in schema_columns]
        if valid_columns:
            analysis["columns"] = valid_columns
            
            # CRITICAL: Ensure all bracketed fields are mapped
            if len(bracketed_mapping) != len(bracketed_fields):
                print(f" ERROR: Not all bracketed fields were mapped!", file=sys.stderr)
                print(f" Expected: {bracketed_fields}, Got mappings for: {list(bracketed_mapping.keys())}", file=sys.stderr)
                return None
            
            # Log the mapping types for debugging
            for bracketed_field, explanation in mapping_explanations.items():
                match_type = "EXACT" if "EXACT" in explanation else "SEMANTIC" if "SEMANTIC" in explanation else "UNKNOWN"
                print(f" Field '{bracketed_field}' mapped with {match_type} matching: {explanation}", file=sys.stderr)
                
            #print(f" Step analysis result for bracketed fields: {analysis}", file=sys.stderr)
            return analysis
        else:
            print(f" No valid columns found in table {table_name}. Requested: {columns}, Available: {schema_columns}", file=sys.stderr)
            return None

    except json.JSONDecodeError as e:
        print(f" Error parsing step analysis JSON: {e}", file=sys.stderr)
        print(f" Raw result: {result}", file=sys.stderr)
        return None
    except Exception as e:
        print(f" Error in step analysis: {e}", file=sys.stderr)
        print(f" Raw result: {result}", file=sys.stderr)
        return None
    
async def validate_tables_for_bracketed_fields_detailed(
    bracketed_fields: List[str], 
    all_schema_fields: Dict[str, List[str]],
    test_step: str = ""
) -> Dict[str, Dict[str, tuple]]:
    """
    COMPLETELY DYNAMIC VERSION: Uses only exact matching and LLM intelligence.
    No hardcoded semantic patterns anywhere.
    Returns: {table_name: {bracketed_field: (column_name, match_type)}}
    """
    if not bracketed_fields or not all_schema_fields:
        return {}

    candidate_tables_with_mappings = {}
    print(f" Dynamic pre-validation for bracketed fields: {bracketed_fields}", file=sys.stderr)

    for table_name, table_columns in all_schema_fields.items():
        table_can_satisfy = True
        table_mappings = {}
        match_confidence_total = 0
        
        for bracketed_field in bracketed_fields:
            field_matched = False
            match_type = None
            matched_column = None
            field_confidence = 0
            
            # STEP 1: Check for EXACT match first
            clean_bracketed = normalize_field_name(bracketed_field)
            
            for column in table_columns:
                clean_column = normalize_field_name(column)
                if clean_bracketed == clean_column:
                    field_matched = True
                    match_type = "EXACT"
                    matched_column = column
                    field_confidence = 100
                    print(f" ✓ EXACT match: '{bracketed_field}' -> '{column}'", file=sys.stderr)
                    break
            
            # STEP 2: Use LLM intelligent selection if no exact match found
            if not field_matched:
                llm_match = await llm_intelligent_field_selection(bracketed_field, table_columns, test_step)
                if llm_match and llm_match.confidence_score >= 70:
                    field_matched = True
                    match_type = "SEMANTIC_INTELLIGENT"
                    matched_column = llm_match.column_name
                    field_confidence = llm_match.confidence_score
            
            # NO STEP 3: No hardcoded fallback - either exact or LLM, nothing else
            
            if field_matched:
                table_mappings[bracketed_field] = (matched_column, match_type)
                match_confidence_total += field_confidence
            else:
                table_can_satisfy = False
                #print(f" ✗ No match found for '{bracketed_field}' in table '{table_name}'", file=sys.stderr)
                break
                
        if table_can_satisfy:
            candidate_tables_with_mappings[table_name] = table_mappings
            avg_confidence = match_confidence_total / len(bracketed_fields) if bracketed_fields else 0
            mapping_details = [f"{field} -> {column} ({match_type})" for field, (column, match_type) in table_mappings.items()]
            #print(f" ✓ Table '{table_name}' satisfies all fields (avg confidence: {avg_confidence:.1f}%): {'; '.join(mapping_details)}", file=sys.stderr)
        else:
            pass
            #print(f" ✗ Table '{table_name}' cannot satisfy all bracketed fields", file=sys.stderr)
    def table_priority(item):
        table_name, mappings = item
        exact_matches = sum(1 for _, (_, match_type) in mappings.items() if match_type == "EXACT")
        intelligent_matches = sum(1 for _, (_, match_type) in mappings.items() if match_type == "SEMANTIC_INTELLIGENT")
        return (exact_matches * 100 + intelligent_matches * 50)
    
    sorted_tables = sorted(candidate_tables_with_mappings.items(), key=table_priority, reverse=True)

    #print(f" Dynamic validation result: {len(sorted_tables)} candidate tables found", file=sys.stderr)
    return dict(sorted_tables)

def validate_tables_for_bracketed_fields(bracketed_fields: List[str], all_schema_fields: Dict[str, List[str]]) -> List[str]:
    """
    UPDATED: Prioritize exact field matches over semantic matches. Only tables that can provide 
    ALL bracketed fields (by exact match first, then semantic match as fallback) are considered.
    This is the original function that returns just table names for backward compatibility.
    """
    detailed_results = validate_tables_for_bracketed_fields_detailed(bracketed_fields, all_schema_fields)
    return list(detailed_results.keys())


def normalize_field_name(field_name: str) -> str:
    """
    NEW HELPER FUNCTION: Normalize field names for exact comparison.
    Converts to lowercase and removes/standardizes spaces, underscores, and dashes.
    """
    if not isinstance(field_name, str):
        return ""
    normalized = field_name.lower().strip()
    # Convert to lowercase and replace separators with underscores, then remove them for comparison
    for sep in [' ', '_', '-']:
        normalized = normalized.replace(sep, '')
    return normalized

#def semantic_field_match(bracketed_field: str, column_name: str) -> bool:
#    """
#    ENHANCED FUNCTION: Check if a bracketed field semantically matches a database column.
#    Now includes better username/name matching and more comprehensive patterns.
#    This should only be used when exact matching fails.
#    """
#    # Clean inputs for semantic comparison
#    bracket_clean = bracketed_field.lower().strip().replace(' ', '').replace('-', '').replace('_', '')
#    column_clean = column_name.lower().strip().replace(' ', '').replace('-', '').replace('_', '')
#    
#    # If this would be an exact match, return False - exact matches should be handled elsewhere
#    if bracket_clean == column_clean:
#        return False  # This should be caught by exact matching logic
#    
#    # ENHANCED semantic patterns for different field types
#    semantic_patterns = {
#        # ID patterns
#        'businessid': ['business_id', 'bus_id', 'company_id', 'org_id'],
#        'userid': ['user_id', 'account_id', 'person_id'],
#        'projectid': ['project_id', 'proj_id'],
#        'clientid': ['client_id', 'customer_id', 'cust_id'],
#        'documentid': ['document_id', 'doc_id', 'file_id'],
#        
#        # ENHANCED Name/Username patterns (CRITICAL FOR YOUR USE CASE)
#        'username': ['user_name', 'login', 'account_name', 'name', 'user', 'login_name'],
#        'user_name': ['username', 'login', 'account_name', 'name'],
#        'login': ['username', 'user_name', 'login_name', 'name'],
#        'name': ['username', 'user_name', 'login', 'full_name', 'display_name'],
#        
#        # Project/Business name patterns  
#        'projectname': ['project_name', 'proj_name', 'project_title'],
#        'companyname': ['company_name', 'business_name', 'organization'],
#        
#        # Date patterns
#        'startdate': ['start_date', 'begin_date', 'from_date'],
#        'enddate': ['end_date', 'finish_date', 'to_date', 'completion_date'],
#        
#        # Status patterns
#        'status': ['state', 'condition', 'stage'],
#        
#        # Contact patterns
#        'email': ['email_address', 'mail', 'e_mail'],
#        'phone': ['phone_number', 'phonenumber', 'telephone', 'mobile', 'contact_number'],
#
#        # Authentication patterns
#        'password': ['pass', 'pwd', 'passcode', 'secret'],
#        
#        # Generic ID fallback
#        'id': ['identifier', 'key', 'reference']
#    }
#    
#    # Check bracketed field against patterns
#    for pattern, possible_columns in semantic_patterns.items():
#        if pattern in bracket_clean:
#            for possible in possible_columns:
#                if possible in column_clean:
#                    print(f" ✓ Semantic match found: {bracketed_field} ~ {column_name} (pattern: {pattern})", file=sys.stderr)
#                    return True
#    
#    # Check reverse - if column has known patterns
#    for pattern, possible_columns in semantic_patterns.items():
#        if pattern in column_clean:
#            for possible in possible_columns:
#                if possible in bracket_clean:
#                    print(f" ✓ Semantic match found: {bracketed_field} ~ {column_name} (reverse pattern: {pattern})", file=sys.stderr)
#                    return True
#    
#    # SPECIAL CASE: Direct semantic relationships for common login fields
#    login_field_mappings = {
#        'username': ['name', 'user', 'login'],
#        'user': ['name', 'username', 'login'], 
#        'login': ['name', 'username', 'user'],
#        'account': ['name', 'username', 'user']
#    }
#    
#    for login_field, related_fields in login_field_mappings.items():
#        if login_field in bracket_clean:
#            for related in related_fields:
#                if related in column_clean:
#                    print(f" ✓ Semantic match found: {bracketed_field} ~ {column_name} (login field mapping)", file=sys.stderr)
#                    return True
#    
#    # Partial containment check (more strict than before)
#    # Only match if one field contains the other and they're both reasonably long
#    if len(bracket_clean) >= 4 and len(column_clean) >= 4:
#        if (bracket_clean in column_clean and len(bracket_clean) > len(column_clean) * 0.6) or \
#           (column_clean in bracket_clean and len(column_clean) > len(bracket_clean) * 0.6):
#            print(f" ✓ Semantic match found: {bracketed_field} ~ {column_name} (partial containment)", file=sys.stderr)
#            return True
#    
#    return False

# Helps Agent think by linking indirect fields(email -> userid -> projectnames)


# Uses WHERE conditions to backtrack and inspect previous results.
# REPLACE THE OLD enhance_where_conditions FUNCTION WITH THIS:


# ------------------------------------------------------------------------------
#Builds SQL queries, Integrates WHERE conditions from previous results using `enhance_where_conditions`.
#Builds SELECT, FROM, JOIN, WHERE, ORDER BY RAND() clauses dynamically
# ------------------------------------------------------------------------------
async def build_sql_for_step(step_analysis: Dict[str, Any], selectional_memory, ws, relationship_graph: Dict[str, Dict[str, List[str]]] = None, memory_key_index: Dict[str, List[Any]] = None) -> str:
    """
    Builds SQL query that ALWAYS uses SELECT * to retrieve complete rows.
    NOW INCLUDES PROPER MEMORY CHECKING ACROSS ALL TABLES and maintains field name mapping.
    """
    import re

    table = step_analysis.get("table", "")
    joins = step_analysis.get("joins", [])
    where_conditions = step_analysis.get("where_conditions", {})
    step_columns = step_analysis.get("columns", [])  # Database column names
    original_action = step_analysis.get("originalAction", "")
    bracketed_fields = step_analysis.get("bracketedFields", [])
    bracketed_mapping = step_analysis.get("bracketed_fields_mapped", {})

    if not table:
        print(" No table specified in step analysis", file=sys.stderr)
        return ""

    print(f" Building SQL for bracketed fields: {bracketed_fields}", file=sys.stderr)
    print(f" Database columns to extract: {step_columns}", file=sys.stderr)
    print(f" Field mapping: {bracketed_mapping}", file=sys.stderr)

    # Get flattened memory
    memory_data = flatten_memory_data(selectional_memory)

    # ENHANCED MEMORY CHECKING - Look across ALL tables and entities for required database columns
    print(f" Checking memory across all entities for required database columns: {step_columns}", file=sys.stderr)
    recent_rows = selectional_memory.get_all()
    
    for i in reversed(range(len(recent_rows))):
        candidate_row = recent_rows[i]
        if isinstance(candidate_row, dict):
            # Check if this row has ALL the required database columns
            if step_columns and all(col in candidate_row for col in step_columns):
                # Prioritize rows from the same table, but allow cross-table reuse
                row_table = candidate_row.get("__table__", "unknown")
                print(f" Memory row [{i+1}] from table [{row_table}] has required columns {step_columns}", file=sys.stderr)
                return f"__USE_MEMORY_ROW__::{i}"

    # ALWAYS use SELECT * to get complete rows
    sql_base = f"SELECT * FROM {table}"

    # Add JOINs if specified
    for join in joins:
        join_table = join.get("table")
        join_on = join.get("on")
        if join_table and join_on:
            sql_base += f" JOIN {join_table} ON {join_on}"
            print(f" Added JOIN: {join_table} ON {join_on}", file=sys.stderr)

    # Build WHERE clause with enhanced condition processing
    enhanced_conditions = await enhance_where_conditions_with_memory(where_conditions, memory_data, table, ws)
    print(f" Enhanced WHERE conditions from context: {enhanced_conditions}", file=sys.stderr)

    relational_constraints = {}
    if relationship_graph and memory_key_index:
        relational_constraints = find_relational_constraints(table, relationship_graph, memory_key_index)
        print(f" Relational constraints found: {relational_constraints}", file=sys.stderr)

    # NEW: Build tiered query strategy with relational consistency
    if enhanced_conditions or relational_constraints:
        tiered_sql = build_tiered_query_strategy(sql_base, enhanced_conditions, relational_constraints, table)
        print(f" Using tiered query strategy: {tiered_sql}", file=sys.stderr)
        return tiered_sql
    else:
        # Fallback to random query (unchanged behavior)
        fallback_sql = sql_base + " ORDER BY RAND() LIMIT 1"
        print(" No contextual or relational conditions found, using random fallback", file=sys.stderr)
        return fallback_sql
    
    # Build context-based SQL with WHERE conditions
#    context_sql = sql_base
#    fallback_sql = sql_base + " ORDER BY RAND() LIMIT 1"  # Fallback query
#    
#    where_parts = []
#    for field, value in enhanced_conditions.items():
#        if value is not None:
#            # Handle different value types properly
#            if isinstance(value, str):
#                # Don't quote numeric strings that should be numbers
#                if value.isdigit():
#                    clause = f"{field} = {value}"
#                else:
#                    # Escape single quotes in string values
#                    escaped_value = value.replace("'", "''")
#                    clause = f"{field} = '{escaped_value}'"
#            elif isinstance(value, (int, float)):
#                clause = f"{field} = {value}"
#            else:
#                clause = f"{field} = '{value}'"
#            
#            where_parts.append(clause)
#            print(f" Added contextual WHERE clause: {clause}", file=sys.stderr)
#
#    if where_parts:
#        context_sql += " WHERE " + " AND ".join(where_parts)
#        context_sql += " LIMIT 1"
#        print(f" Context-based SQL: {context_sql}", file=sys.stderr)
#        
#        # Return context SQL with fallback info
#        return f"__CONTEXT_WITH_FALLBACK__::{context_sql}::{fallback_sql}"
#    else:
#        print(" No contextual WHERE conditions found, using fallback query", file=sys.stderr)
#        return fallback_sql

def infer_explicit_fields_from_action(action_text: str, memory_data: Dict[str, Any]) -> List[str]:
    """Infer relevant fields from action text if LLM returns '*'."""
    action_text = action_text.lower()
    keywords = ["email", "password", "id", "name", "phone", "status", "code"]

    inferred_fields = []

    for field in memory_data.keys():
        for keyword in keywords:
            if keyword in field.lower() and keyword in action_text:
                inferred_fields.append(field)
                break  # Avoid duplicates

    return inferred_fields or ["*"]

async def process_selectional_testcase(ws, tc: Dict[str, Any], selectional_memory,  relationship_graph: Dict[str, Dict[str, List[str]]] = None, memory_key_index: Dict[str, List[Any]] = None) -> List[Dict[str, Any]]:
    """
    Process selectional test case and return ordered list of field-value pairs.
    NOW PROPERLY USES BRACKETED FIELD NAMES in output and handles cross-table memory reuse.
    INCLUDES FALLBACK: If any forced_creational_fields exist, also generate them at the end.
    """
    ordered_results = []
    tc_id = tc.get("tcSeqNo", tc.get("tcId", "unknown"))
    print(f" Processing selectional test case: {tc_id}", file=sys.stderr)

    if memory_key_index is None:
        memory_key_index = {}

    # Get step execution order and identify selectional steps
    step_order_map = extract_step_order_from_testcase(tc)
    selectional_steps = await identify_selectional_steps(tc, selectional_memory)
    if not selectional_steps:
        print(" No selectional steps with <<field>> patterns identified", file=sys.stderr)
        # --- CREATIONAL FALLBACK PATCH: still check for forced_creational_fields even if no selectional steps ---
        fallback_results = []
        forced_fields = tc.get("forced_creational_fields", [])
        if forced_fields:
            print(f" Forced creational fields present even in selectional test case: {forced_fields}", file=sys.stderr)
            fake_data = await generate_fake_data_for_fields(tc, forced_fields)
            max_step = max(step_order_map.keys()) if step_order_map else 0
            creational_step_num = max_step + 1
            for field_name, field_value in fake_data.items():
                fallback_results.append({
                    "fieldName": field_name,
                    "value": field_value,
                    "step_number": creational_step_num
                })
        # Return only fallback results if no selectional steps
        return fallback_results

    # Map to collect each step's results
    step_results: Dict[int, List[Dict[str, Any]]] = {}

    for i, step_analysis in enumerate(selectional_steps):
        step_num = step_analysis.get("stepNumber", i + 1)
        original_action = step_analysis.get("originalAction", "")
        db_columns = step_analysis.get("columns", [])  # Database column names
        bracketed_fields = step_analysis.get("bracketedFields", [])  # Original bracketed field names
        bracketed_mapping = step_analysis.get("bracketed_fields_mapped", {})  # Mapping dict

        print(f"\n=== Processing Step {step_num} ===", file=sys.stderr)
        print(f"Action: {original_action}", file=sys.stderr)
        print(f"Bracketed fields: {bracketed_fields}", file=sys.stderr)
        print(f"Database columns: {db_columns}", file=sys.stderr)
        print(f"Field mapping: {bracketed_mapping}", file=sys.stderr)

        debug_memory_state(selectional_memory, f"Before Step {step_num}")
        sql = await build_sql_for_step(step_analysis, selectional_memory, ws, relationship_graph, memory_key_index)
        if not sql:
            print(f" Could not generate SQL for step {step_num}", file=sys.stderr)
            continue

        # 1) Memory‐reuse branch
        if sql.startswith("__USE_MEMORY_ROW__"):
            parts = sql.split("::")
            reuse_index = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else -1
            all_rows = selectional_memory.get_all()

            if all_rows and 0 <= reuse_index < len(all_rows):
                complete_row = all_rows[reuse_index]
                step_field_data: List[Dict[str, Any]] = []

                # Use bracketed field names for output, but extract from database columns
                for bracketed_field in bracketed_fields:
                    # Find the corresponding database column name
                    db_column = bracketed_mapping.get(bracketed_field, bracketed_field)
                    
                    if db_column in complete_row:
                        raw = complete_row[db_column]
                        transformed = transform_value_based_on_action(str(raw), original_action)
                        step_field_data.append({
                            "fieldName": bracketed_field,  # Use original bracketed field name
                            "value": transformed,
                            "step_number": step_num
                        })
                        print(f" Extracted from memory: {bracketed_field} (from db column {db_column}) = {transformed}", file=sys.stderr)
                    else:
                        print(f" Database column '{db_column}' for bracketed field '{bracketed_field}' not found in memory row", file=sys.stderr)

                step_results[step_num] = step_field_data
                print(f" Used memory row [{reuse_index+1}] for Step {step_num} (no SQL executed)", file=sys.stderr)
            else:
                print(f" Could not find valid memory row at index {reuse_index}, skipping", file=sys.stderr)

            continue

        # 2) SQL‐execution branch (now handles context-with-fallback)
        print(f" Executing SQL (with potential fallback): {sql}", file=sys.stderr)
        try:
            # The execute_sql_query function now handles context-with-fallback automatically
            rows = await execute_sql_query(ws, sql)
            
            if not rows:
                print(f" Step {step_num} returned no data even after fallback attempts", file=sys.stderr)
                continue

            complete_row = rows[0]
            print(f" Retrieved complete row with {len(complete_row)} fields", file=sys.stderr)

            step_field_data = []
            # Use bracketed field names for output, but extract from database columns
            for bracketed_field in bracketed_fields:
                # Find the corresponding database column name
                db_column = bracketed_mapping.get(bracketed_field, bracketed_field)
                
                if db_column in complete_row:
                    raw = complete_row[db_column]
                    transformed = transform_value_based_on_action(str(raw), original_action)
                    step_field_data.append({
                        "fieldName": bracketed_field,  # Use original bracketed field name
                        "value": transformed,
                        "step_number": step_num
                    })
                    print(f" Extracted for output: {bracketed_field} (from db column {db_column}) = {transformed}", file=sys.stderr)
                else:
                    print(f" Database column '{db_column}' for bracketed field '{bracketed_field}' not found in retrieved row", file=sys.stderr)

            step_results[step_num] = step_field_data

            # Always store the complete row in memory
            table_name = step_analysis.get("table", "unknown")
            selectional_memory.add(complete_row, table_name)
            update_memory_key_index(complete_row, memory_key_index)

            print(f" Stored complete row in memory from table [{table_name}] ({len(complete_row)} fields)", file=sys.stderr)
            debug_memory_state(selectional_memory, f"After Step {step_num}")

        except Exception as e:
            print(f" Error executing step {step_num}: {e}", file=sys.stderr)
            continue

    # Build ordered_results preserving original step order
    for num in sorted(step_results.keys()):
        ordered_results.extend(step_results[num])

    # --- CREATIONAL FALLBACK PATCH: handle any forced_creational_fields after all selectional steps ---
    fallback_results = []
    forced_fields = tc.get("forced_creational_fields", [])
    if forced_fields:
        print(f" Forced creational fields present in selectional test case: {forced_fields}", file=sys.stderr)
        fake_data = await generate_fake_data_for_fields(tc, forced_fields)
        max_step = max(step_order_map.keys()) if step_order_map else 0
        creational_step_num = max_step + 1
        for field_name, field_value in fake_data.items():
            fallback_results.append({
                "fieldName": field_name,
                "value": field_value,
                "step_number": creational_step_num
            })

    # Merge fallback results after all ordered_results
    ordered_results.extend(fallback_results)

    print(f"\n Selectional processing complete.", file=sys.stderr)
    return ordered_results
# ---------------------------------------------------------------------------
# DYNAMIC CLASSIFICATION AND ANALYSIS
# ---------------------------------------------------------------------------
async def classify_tc(tc: Dict[str, Any]) -> str:
    """Classify test case using LLM analysis with consistent field detection"""
    prompt = f"""
    You are a test case classifier that must provide CONSISTENT results.

Your goal is to classify the test case as one of the following based on ACTION KEYWORDS:
- "selectional" → if the test case primarily involves retrieving/accessing existing data
- "creational" → if the test case primarily involves creating/generating new data  
- "hybrid" → if it involves a mix of both operations

 KEYWORD-BASED CLASSIFICATION RULES:

**CREATIONAL KEYWORDS** (classify as "creational"):
- create, modify, add, generate, insert, build, construct, update
- submit, save, register, establish, define, fill (when creating NEW data)

**SELECTIONAL KEYWORDS** (classify as "selectional"):  
- select, retrieve, check, lookup, fetch, get, find, search, view, display
- validate, confirm, examine, review, navigate, click, login
- enter (when using EXISTING credentials/data like "enter valid username")

 CLASSIFICATION LOGIC:
1. **Scan ALL action steps** for these keywords
2. **Context-aware "enter" handling**:
   - "enter valid/existing [credentials/data]" → SELECTIONAL
   - "enter new/custom [data]" → CREATIONAL
3. **Count the predominant keyword type**:
   - If majority are CREATIONAL keywords → return "creational"
   - If majority are SELECTIONAL keywords → return "selectional" 
   - If roughly equal mix → return "hybrid"

4. **Priority order**: Look at the main objective of the test case. What is the primary goal?

 EXAMPLES:
- "Enter valid username and password" → selectional (using existing credentials)
- "Create new user account" → creational
- "Enter custom project name" → creational
- "Select user and update profile" → hybrid

Return exactly one word: selectional, creational, or hybrid

Test case:
{safe_dump(tc)}

Return exactly one word: selectional, creational, or hybrid
"""
    
    result = await ask_openai(prompt)
    result = result.strip().lower()
    
    # Validate result
    if result not in ["selectional", "creational", "hybrid"]:
        print(f" Invalid classification '{result}', defaulting to 'hybrid'", file=sys.stderr)
        result = "hybrid"
    
    print(f" Classified as: {result}", file=sys.stderr)
    return result


async def analyze_fields_for_tc(tc: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    UPDATED: Analyze test case fields with DATABASE-FIRST approach using LLM semantic understanding.
    Fields that exist in database schema (exact or LLM semantic match) are classified as selectional.
    Only fields with NO database matches are classified as creational.
    """
    import re

    step_actions = []
    step_results = []
    step_array = tc.get("stepArray", [])
    all_bracketed_fields = []
    bracketed_field_contexts = {}
    bracketed_pattern = r'<<([^>]+)>>'

    for step in step_array:
        tc_step = step.get("tcStep", "")
        tc_result = step.get("tcResult", "")

        if tc_step.strip():
            step_actions.append(tc_step.strip())
            fields_in_step = re.findall(bracketed_pattern, tc_step)
            for field in fields_in_step:
                all_bracketed_fields.append(field)
                bracketed_field_contexts[field] = tc_step.strip()
        if tc_result.strip():
            step_results.append(tc_result.strip())

    if not step_actions or not all_bracketed_fields:
        print(" No tcStep actions or <<field>> patterns found in test case", file=sys.stderr)
        return {"selectional": [], "creational": []}

    print(f" ------Found bracketed fields in test case-----: {all_bracketed_fields}", file=sys.stderr)
    print(f" Field contexts: {bracketed_field_contexts}", file=sys.stderr)

    # DATABASE-FIRST CLASSIFICATION: Check schema availability first
    if not DB_SCHEMA:
        print(" DB_SCHEMA not available for field classification, falling back to action-based analysis", file=sys.stderr)
        # Fallback to old behavior if schema not available
    else:
        # Get all schema fields for comprehensive checking
        all_schema_fields = {}
        for table_name, table_info in DB_SCHEMA.items():
            table_fields = extract_schema_columns(table_name)
            all_schema_fields[table_name] = table_fields
        
        # Classify each bracketed field based on database availability
        selectional_fields = []
        creational_fields = []
        
        for field in all_bracketed_fields:
            has_database_match = False
            match_details = []
            
            # Check for exact match across all tables
            for table_name, table_columns in all_schema_fields.items():
                for column in table_columns:
                    if normalize_field_name(field) == normalize_field_name(column):
                        has_database_match = True
                        match_details.append(f"EXACT match: {column} in {table_name}")
                        break
                if has_database_match:
                    break
            
            # If no exact match, use LLM intelligent matching
            if not has_database_match:
                for table_name, table_columns in all_schema_fields.items():
                    # Use the field context for better LLM understanding
                    field_context = bracketed_field_contexts.get(field, "")
                    llm_match = await llm_intelligent_field_selection(field, table_columns, field_context)
                    if llm_match and llm_match.confidence_score >= 70:
                        has_database_match = True
                        match_details.append(f"LLM SEMANTIC match: {llm_match.column_name} in {table_name} ({llm_match.confidence_score}%)")
                        break
            
            # Classify based on database availability
            if has_database_match:
                selectional_fields.append(field)
                print(f" Field '{field}' classified as SELECTIONAL - {'; '.join(match_details)}", file=sys.stderr)
            else:
                creational_fields.append(field)
                print(f" Field '{field}' classified as CREATIONAL - No database matches found", file=sys.stderr)
        
        # Handle forced creational fields (override database-first logic)
        forced_fields = set(tc.get("forced_creational_fields", []))
        if forced_fields:
            print(f" Applying forced_creational_fields override: {forced_fields}", file=sys.stderr)
            # Remove from selectional and add to creational
            selectional_fields = [f for f in selectional_fields if f not in forced_fields]
            for f in forced_fields:
                if f not in creational_fields:
                    creational_fields.append(f)
        
        #print(f" DATABASE-FIRST field analysis - Selectional: {selectional_fields}, Creational: {creational_fields}", file=sys.stderr)
        return {"selectional": selectional_fields, "creational": creational_fields}

    # FALLBACK: Original action-based analysis if schema not available
    tc_description = tc.get("tcDescription", tc.get("title", ""))

    prompt = f"""
You are a master test‑case field analyzer that ONLY analyzes <<field>> patterns with ACTION-BASED CLASSIFICATION.
This is a FALLBACK analysis when database schema is not available.

 GOAL: Classify each bracketed field as either "selectional" or "creational" based on the ACTION VERBS in their step context.

 **ENHANCED CLASSIFICATION RULES:**

**CREATIONAL ACTION VERBS** → classify <<field>> as "creational":
- **generate, create, build, construct, design, develop, compose, write, make**
- **add, insert, submit, save, register, establish, define, fill** (when creating NEW data)
- **customize, configure, input** (when providing new data)

**SELECTIONAL ACTION VERBS** → classify <<field>> as "selectional":
- **enter, input, type** (when using EXISTING credentials/data like username/password)
- **select, choose, pick, retrieve, get, fetch, find, search, view, display**
- **validate, confirm, examine, review, navigate, click, login, access**

 **CRITICAL CLASSIFICATION LOGIC:**

1. **PRIMARY VERB ANALYSIS**:
   - Look at the MAIN ACTION VERB in each step containing a <<field>>
   - If step says "Generate <<field>>" → ALWAYS creational
   - If step says "Create <<field>>" → ALWAYS creational
   - If step says "Enter <<username>>" → typically selectional (existing credentials)
   - If step says "Enter <<password>>" → typically selectional (existing credentials)

2. **CONTEXT-AWARE EXCEPTIONS**:
   - "Enter <<project_name>>" with creative context → creational
   - "Generate <<anything>>" → ALWAYS creational
   - "Create <<anything>>" → ALWAYS creational
   - "Design <<anything>>" → ALWAYS creational

3. **AUTHENTICATION VS CREATION**:
   - Username/password for login → selectional
   - Custom project data, descriptions, names → creational
   - Configuration or new content → creational

 **FIELD-BY-FIELD ANALYSIS:**
Analyze each bracketed field individually based on its step context:

{chr(10).join([f"- Field: '{field}' in step: '{context}'" for field, context in bracketed_field_contexts.items()])}

 **EXAMPLES FOR CLARITY:**
- "Enter <<username>>" → selectional (existing login credential)
- "Enter <<password>>" → selectional (existing login credential)  
- "Generate <<project description>>" → creational (generating new content)
- "Create <<project name>>" → creational (creating new data)
- "Input <<custom description>>" → creational (providing new content)

 **OUTPUT FORMAT:**
Return only valid JSON with ONLY the bracketed fields classified:
{{
    "selectional": ["BRACKETED_FIELD1", "BRACKETED_FIELD2"],
    "creational": ["BRACKETED_FIELD3", "BRACKETED_FIELD4"]  
}}

 **ANALYZE THESE SPECIFIC BRACKETED FIELDS:**
Bracketed Fields to Classify: {all_bracketed_fields}

Field Contexts:
{chr(10).join([f"'{field}': '{context}'" for field, context in bracketed_field_contexts.items()])}

Test Case Description: {tc_description}

 **IMPORTANT**: Focus on the ACTION VERB in each step. "Generate" and "Create" should ALWAYS result in creational classification.

Database schema (for reference only):
{safe_dump(DB_SCHEMA)}
"""

    result = await ask_openai(prompt)

    # Clean up markdown if present
    if result.startswith("```"):
        result = "\n".join(line for line in result.splitlines() if not line.strip().startswith("```"))

    try:
        field_analysis = json.loads(result)
        if not isinstance(field_analysis, dict):
            raise ValueError("Expected JSON object")

        selectional = set(field_analysis.get("selectional", []))
        creational = set(field_analysis.get("creational", []))

        # ----- DYNAMIC PATCH: include forced_creational_fields -----
        forced_fields = set(tc.get("forced_creational_fields", []))
        if forced_fields:
            print(f" Adding forced_creational_fields to creational: {forced_fields}", file=sys.stderr)
        # Remove from selectional if present, then add to creational
        selectional -= forced_fields
        creational |= forced_fields
        # ----------------------------------------------------------

        # Remove any bracketed fields not found in this test case (safety)
        all_found = set(all_bracketed_fields)
        selectional = [f for f in selectional if f in all_found]
        creational = [f for f in creational if f in all_found]

        print(f" FALLBACK bracketed field analysis - Selectional: {selectional}, Creational: {creational}", file=sys.stderr)
        return {"selectional": selectional, "creational": creational}

    except Exception as e:
        print(f" Error parsing field analysis: {e}", file=sys.stderr)
        # Fallback: Auto-classify forced fields as creational
        selectional = []
        creational = list(tc.get("forced_creational_fields", []))
        return {"selectional": selectional, "creational": creational}
        

def validate_schema_fields(fields: List[str]) -> List[str]:
    """Validate that requested fields exist in database schema"""
    if not fields or not DB_SCHEMA:
        return fields
    
    # Extract all column names from schema
    valid_columns = set()
    for table_name, table_info in DB_SCHEMA.items():
        if isinstance(table_info, dict) and "columns" in table_info:
            for column_info in table_info["columns"]:
                if isinstance(column_info, dict) and "name" in column_info:
                    column_name = column_info["name"]
                    valid_columns.add(column_name)
                    valid_columns.add(f"{table_name}.{column_name}")
    
    # Keep fields that exist in schema
    validated_fields = []
    for field in fields:
        if (field in valid_columns or 
            any(field.lower() in col.lower() for col in valid_columns) or
            any(col.lower() in field.lower() for col in valid_columns)):
            validated_fields.append(field)
        else:
            print(f"  Field '{field}' not found in schema", file=sys.stderr)
    
    return validated_fields

# ---------------------------------------------------------------------------
# DYNAMIC DATA GENERATION
# ---------------------------------------------------------------------------
fake = Faker()

async def generate_fake_data_for_fields(tc: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """Generate realistic, unique, and contextually appropriate fake data for specified fields"""
    
    if not fields:
        return {}

    entropy_seed = str(uuid.uuid4())
    entropy_snippet = fake.paragraph(nb_sentences=2)
    random_words = ", ".join(fake.words(nb=5))

    prompt = f"""
You are an advanced QA assistant trained to create **realistic, high-utility test data** for diverse software applications.

 Objective:
Generate values for the specified fields based on the test case provided.
- If any field values are explicitly mentioned in the test case, **preserve them exactly**.
- For all other fields, generate **original, rich, and context-aware data** that feels natural and believable.

 Instructions:
1. Read the test case step(s) and understand any implied **theme, context, or tone**.
2. If the context hints at a certain domain or technology, adapt the data accordingly — **but only when it’s clearly indicated**.
3. If **no specific context is provided**, take full creative liberty and invent **highly realistic, imaginative values**. Think outside the box.
4. Avoid boring, generic filler terms (like “Platform”, “Initiative”, “System”) unless the test case uses them directly.
5. Every output must be **fresh**, **non-repetitive**, and **human-like**. Avoid overused patterns or robotic phrasing.
6. Data should be practical and suitable for enterprise-level test environments — no humor, slang, or fake-looking names.
7. Output only a **valid JSON** object — the keys must match the provided `fields` list exactly.

 Seed of randomness: {entropy_seed}
 Creative entropy input: {entropy_snippet}
 Free association terms: {random_words}

 Test Case Details:
{json.dumps(tc, indent=2)}

 Fields to Generate:
{fields}

 Output Format:
{{
  "field1": "value1",
  "field2": "value2",
  ...
}}
"""
    result = await ask_openai(prompt, temperature=0.9)

    # Clean up triple backticks if the model wraps response in a code block
    if result.startswith("```"):
        result = "\n".join(line for line in result.splitlines() if not line.strip().startswith("```"))

    try:
        fake_data = json.loads(result)
        if not isinstance(fake_data, dict):
            raise ValueError("Expected JSON object")

        #  Only keep keys that are in the requested fields
        filtered_data = {k: v for k, v in fake_data.items() if k in fields}

        print(f" Generated fake data: {filtered_data}", file=sys.stderr)
        return filtered_data

    except Exception as e:
        print(f"  Error generating fake data: {e}", file=sys.stderr)
        print(result, file=sys.stderr)
        return {}

async def fix_failed_sql(sql: str, step_analysis: Dict[str, Any], error: str) -> str:
    """
    Fix failed SQL queries with LLM using step context and schema.
    Cleans up formatting artifacts from LLM responses.
    """
    prompt = f"""
This SQL query failed. Fix it using the schema and step context.

Failed SQL:
{sql}

Error:
{error}

Step context:
{safe_dump(step_analysis)}

Database schema:
{safe_dump(DB_SCHEMA)}

Instructions:
- Return only the corrected SQL query.
- Do NOT include any markdown formatting (like ```).
- Do NOT include explanations or comments.
- Ensure WHERE conditions use real values from context.
- Use LIMIT 1 if selecting a single row.
- Fix any typos in SQL keywords (ORDER BY, WHERE, etc.).
- Use proper table and column names from the schema.
"""

    result = await ask_openai(prompt)

    #  Cleanup step - remove quotes and formatting artifacts
    fixed_sql = result.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')

    # Remove markdown formatting
    if "```sql" in fixed_sql:
        fixed_sql = fixed_sql.split("```sql")[-1].split("```")[0].strip()
    elif "```" in fixed_sql:
        # Get content between first ``` blocks
        parts = fixed_sql.split("```")
        if len(parts) >= 2:
            fixed_sql = parts[1].strip()
        else:
            fixed_sql = parts[0].strip()

    # Remove trailing semicolon and extra whitespace
    fixed_sql = fixed_sql.rstrip(";").strip()
    
    # Remove common typos
    fixed_sql = fixed_sql.replace("ORDER BYY", "ORDER BY")
    fixed_sql = fixed_sql.replace("WHERRE", "WHERE")
    fixed_sql = fixed_sql.replace("SELECTT", "SELECT")
    fixed_sql = fixed_sql.replace("FROMM", "FROM")
    
    # Ensure single spaces around keywords
    import re
    fixed_sql = re.sub(r'\s+', ' ', fixed_sql)

    print(f" Fixed SQL: {fixed_sql}", file=sys.stderr)
    return fixed_sql

# ---------------------------------------------------------------------------
# UNIVERSAL PROCESSORS
# ---------------------------------------------------------------------------
async def process_creational_testcase(ws, tc: Dict[str, Any], creational_memory) -> List[Dict[str, Any]]:
    """Process creational test case and return ordered list of field-value pairs"""
    ordered_results = []
    
    field_analysis = await analyze_fields_for_tc(tc)
    creational_fields = field_analysis["creational"]

    if not creational_fields:
        print("  No creational fields found", file=sys.stderr)
        return ordered_results

    # Get step execution order to determine where creational fields should appear
    step_order_map = extract_step_order_from_testcase(tc)
    
    # Generate unique fake data (not already in memory)
    max_attempts = 5
    attempt = 0
    
    # Get existing data from memory for deduplication
    existing_entities = creational_memory.get_all()
    print(f" Existing entities in creational memory: {len(existing_entities)}", file=sys.stderr)
    
    fake_data = await generate_fake_data_for_fields(tc, creational_fields)
    
    # Check for duplicates in memory
    while attempt < max_attempts:
        is_duplicate = False
        for existing_entity in existing_entities:
            if isinstance(existing_entity, dict) and isinstance(fake_data, dict):
                # Check if any key-value pairs match
                common_keys = set(existing_entity.keys()) & set(fake_data.keys())
                if common_keys:
                    matches = sum(1 for key in common_keys if existing_entity.get(key) == fake_data.get(key))
                    if matches > len(common_keys) * 0.5:  # More than 50% match
                        is_duplicate = True
                        break
        
        if not is_duplicate:
            break
            
        print(f"Duplicate fake data detected (attempt {attempt + 1}), regenerating...", file=sys.stderr)
        fake_data = await generate_fake_data_for_fields(tc, creational_fields)
        attempt += 1
    
    # Store generated data in creational memory
    if fake_data:
        creational_memory.add(fake_data)
        print(f" Added new entity to creational memory. Total entities: {len(creational_memory.get_all())}", file=sys.stderr)
    
    # Convert fake data to ordered field-value format
    # For creational data, we'll assign it to the last step + 1 for ordering
    max_step = max(step_order_map.keys()) if step_order_map else 0
    creational_step_num = max_step + 1
    
    for field_name, field_value in fake_data.items():
        ordered_results.append({
            "fieldName": field_name,
            "value": field_value,
            "step_number": creational_step_num
        })
        print(f" Added creational field: {field_name} = {field_value}", file=sys.stderr)
    
    return ordered_results

async def process_universal_hybrid(ws, tc: Dict[str, Any], selectional_memory, creational_memory, relationship_graph: Dict[str, Dict[str, List[str]]] = None, memory_key_index: Dict[str, List[Any]] = None) -> List[Dict[str, Any]]:
    """
    Universal hybrid processor that maintains execution order and prevents duplicate processing.
    NOW ENSURES NO OVERLAP between selectional and creational field processing.
    """
    ordered_results = []
    
    if memory_key_index is None:
        memory_key_index = {}

    # Get consistent field analysis
    field_analysis = await analyze_fields_for_tc(tc)
    selectional_fields = field_analysis["selectional"]
    creational_fields = field_analysis["creational"]

    print(f" Processing hybrid test case with {len(selectional_fields)} selectional and {len(creational_fields)} creational fields", file=sys.stderr)
    print(f" Selectional fields: {selectional_fields}", file=sys.stderr)
    print(f" Creational fields: {creational_fields}", file=sys.stderr)

    # Get step execution order
    step_order_map = extract_step_order_from_testcase(tc)

    # Process selectional fields and get their results
    selectional_results = []
    if selectional_fields:
        print(" Processing selectional fields...", file=sys.stderr)
        selectional_results = await process_selectional_testcase(ws, tc, selectional_memory, relationship_graph, memory_key_index)

    # Process creational fields and get their results (ONLY if not already processed)
    creational_results = []
    if creational_fields:
        print(" Processing creational fields...", file=sys.stderr)
        
        # Generate fake data only for creational fields
        fake_data = await generate_fake_data_for_fields(tc, creational_fields)
        
        if fake_data:
            # Store in creational memory
            creational_memory.add(fake_data)
            print(f" Added creational data to memory: {fake_data}", file=sys.stderr)
            
            # Convert to result format
            max_step = max(step_order_map.keys()) if step_order_map else 0
            creational_step_num = max_step + 1
            
            for field_name, field_value in fake_data.items():
                if field_name in creational_fields:  # Only include classified creational fields
                    creational_results.append({
                        "fieldName": field_name,
                        "value": field_value,
                        "step_number": creational_step_num
                    })
                    print(f" Added creational field: {field_name} = {field_value}", file=sys.stderr)

    # Combine results while maintaining execution order
    all_results = selectional_results + creational_results
    
    # Sort by step number to maintain execution order
    ordered_results = sorted(all_results, key=lambda x: x.get("step_number", 999))
    
    # Remove step_number from final output as it's only used for ordering
    for result in ordered_results:
        if "step_number" in result:
            del result["step_number"]

    # Validate no duplicates
    seen_fields = set()
    final_results = []
    for result in ordered_results:
        field_name = result.get("fieldName")
        if field_name not in seen_fields:
            final_results.append(result)
            seen_fields.add(field_name)
        else:
            print(f" Removing duplicate field: {field_name}", file=sys.stderr)

    print(f" Hybrid processing complete. Final unique fields: {len(final_results)}", file=sys.stderr)
    return final_results

# ---------------------------------------------------------------------------
# NEW JSON OUTPUT FORMATTING FUNCTION
# ---------------------------------------------------------------------------

#def print_json_output(test_data_list: List[Dict[str, Any]]) -> None:
#    """
#    Print clean JSON output in the exact required format to terminal.
#    Handles empty data gracefully and ensures proper JSON structure.
#    """
#    
#    if not test_data_list:
#        print(" No output data to display", file=sys.stderr)
#        # Still output valid JSON structure even if empty
#        output_json = {"TestData": []}
#    else:
#        # Build the JSON structure exactly as requested
#        test_data_entries = []
#        
#        for item in test_data_list:
#            if isinstance(item, dict) and "fieldName" in item and "value" in item:
#                # Ensure both fieldName and value are not None/empty
#                field_name = item["fieldName"]
#                field_value = item["value"]
#                
#                if field_name is not None and field_value is not None:
#                    test_data_entries.append({
#                        "fieldName": str(field_name),
#                        "value": str(field_value)
#                    })
#        
#        output_json = {"TestData": test_data_entries}
#    
#    # Print JSON to stdout (main output) and log to stderr
#    json_output = json.dumps(output_json, indent=2, ensure_ascii=False)
#    print(json_output)  # Main output to stdout
    
    # Log summary to stderr for debugging
   #print(f" JSON Output Generated: {len(output_json['TestData'])} field entries", file=sys.stderr)
   #for i, entry in enumerate(output_json['TestData'], 1):
   #    print(f"  [{i}] {entry['fieldName']}: {entry['value']}", file=sys.stderr)
# ---------------------------------------------------------------------------
# OUTPUT FORMATTING
# ---------------------------------------------------------------------------
def extract_step_order_from_testcase(tc: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """Extract step execution order from test case stepArray with step numbers"""
    step_array = tc.get("stepArray", [])
    step_order_map = {}
    
    for index, step in enumerate(step_array):
        step_number = index + 1  # 1-based indexing
        step_order_map[step_number] = {
            "action": step.get("tcStep", ""),
            "result": step.get("tcResult", ""),
            "index": index,
            "step_number": step_number
        }
    
    print(f" Extracted {len(step_order_map)} steps from test case", file=sys.stderr)
    return step_order_map

def print_json_output(test_data_list: List[Dict[str, Any]]) -> None:
    """
    Print clean JSON output in the exact required format to terminal.
    Handles empty data gracefully and ensures proper JSON structure.
    """
    
    if not test_data_list:
        print(" No output data to display", file=sys.stderr)
        # Still output valid JSON structure even if empty
        output_json = {"TestData": []}
    else:
        # Build the JSON structure exactly as requested
        test_data_entries = []
        
        for item in test_data_list:
            if isinstance(item, dict) and "fieldName" in item and "value" in item:
                # Ensure both fieldName and value are not None/empty
                field_name = item["fieldName"]
                field_value = item["value"]
                
                if field_name is not None and field_value is not None:
                    test_data_entries.append({
                        "fieldName": str(field_name),
                        "value": str(field_value)
                    })
        
        output_json = {"TestData": test_data_entries}
    
    # Print JSON to stdout (main output) and log to stderr
    json_output = json.dumps(output_json, indent=2, ensure_ascii=False)
    print(json_output)  # Main output to stdout
    
    # Log summary to stderr for debugging
    #print(f" JSON Output Generated: {len(output_json['TestData'])} field entries", file=sys.stderr)
    #for i, entry in enumerate(output_json['TestData'], 1):
    #    print(f"  [{i}] {entry['fieldName']}: {entry['value']}", file=sys.stderr)

   
# ---------------------------------------------------------------------------
# ENHANCED OUTPUT FORMATTING
# ---------------------------------------------------------------------------

async def validate_generated_data(data: Dict[str, Any], tc: Dict[str, Any]) -> bool:
    """Validate generated data against test case requirements"""
    if not data:
        return False

    try:
        # Check date fields if present
        if "projectStartDate" in data and "projectEndDate" in data:
            from datetime import datetime
            start_date = datetime.strptime(data["projectStartDate"], "%Y-%m-%d")
            end_date = datetime.strptime(data["projectEndDate"], "%Y-%m-%d")
            if (end_date - start_date).days < 3:
                print(f" Date validation failed: end date must be at least 3 days after start date", file=sys.stderr)
                return False
        # Check for empty string values
        for key, value in data.items():
            if value == "" or value is None:
                print(f" Validation failed: field '{key}' is empty or None", file=sys.stderr)
                return False
        return True
    except Exception as e:
        print(f" Data validation error: {e}", file=sys.stderr)
        return False
# ---------------------------------------------------------------------------
# MEMORY SETUP (LangGraph or placeholder)
# ---------------------------------------------------------------------------
# If you have LangGraph installed, import its memory class here:
# from langgraph.memory import EntityMemory

# For now, let's use a simple placeholder class:
class SimpleMemory:
    def __init__(self):
        self.entities = []

    def add(self, entity: Dict[str, Any], table_name: str = None):
        """Add entity with optional table context"""
        if table_name:
            entity_with_table = {"__table__": table_name, **entity}
        else:
            entity_with_table = entity  # fallback if context not available

        # Deduplicate by content
        if entity_with_table not in self.entities:
            self.entities.append(entity_with_table)

    def get_all(self):
        return self.entities

    def clear(self):
        self.entities = []

@dataclass
class FieldMatch:
    column_name: str
    confidence_score: int  # 0-100
    reasoning: str
    match_type: str  # "EXACT" or "SEMANTIC_INTELLIGENT"
    
async def llm_intelligent_field_selection(
    bracketed_field: str,
    available_columns: List[str], 
    test_step: str
) -> Optional[FieldMatch]:
    """
    Completely dynamic LLM-powered field selection with no hardcoded patterns.
    LLM analyzes semantic meaning and context without any predefined rules.
    """
    
    if not available_columns:
        return None
    
    # Let LLM handle ALL semantic understanding dynamically
    prompt = f"""
You are an intelligent field mapping system. Analyze the bracketed field and its context to select the best database field match.

Test Step: "{test_step}"
Bracketed Field: "<<{bracketed_field}>>"
Available Database Fields: {available_columns}

Your Task:
1. **Understand the bracketed field**: What does "<<{bracketed_field}>>" represent? What type of data or concept does it refer to?

2. **Analyze the surrounding context**: Look at the test step text around "<<{bracketed_field}>>". Does the context provide clues about what specific type of field this is? Ignore generic words like "field", "input", "textbox" and focus on meaningful business terms.

3. **Understand database fields**: For each available database field, understand what type of data it likely stores based on its name.

4. **Find the best semantic match**: Select the database field that best represents the same concept as "<<{bracketed_field}>>" considering both the field name and any context clues.

5. **Confidence assessment**: How confident are you that this is the correct mapping? (0-100%)

Examples of good semantic matching:
- "<<license>>" in context "enter license in the field" → "license_number" (license refers to license identifier)
- "<<holder>>" in context "select holder from dropdown" → "holder_name" (holder refers to person's name)
- "<<plate>>" in context "vehicle registration" → "license_plate" (plate refers to vehicle license plate)

Return your analysis in this exact JSON format:
{{
  "column_name": "best_matching_database_field",
  "confidence_score": 85,
  "reasoning": "Detailed explanation of why this field was selected, including field meaning analysis and context considerations",
  "field_understanding": "What the bracketed field represents",
  "context_analysis": "How the surrounding context influenced the decision",
  "match_type": "SEMANTIC_INTELLIGENT"
}}

If no good semantic match exists (confidence < 70), return: null

Be precise and analytical in your reasoning. Focus on semantic meaning rather than superficial name similarity.
"""
    
    try:
        result = await ask_openai(prompt)
        
        # Clean up markdown if present
        if result.startswith("```"):
            result = "\n".join(line for line in result.splitlines() if not line.strip().startswith("```"))
        
        if result.strip().lower() == "null":
            print(f" LLM found no good semantic match for '{bracketed_field}' in context: {test_step}", file=sys.stderr)
            return None
        
        match_data = json.loads(result)
        
        # Validate the response
        if not isinstance(match_data, dict):
            print(f" Invalid LLM response format for '{bracketed_field}'", file=sys.stderr)
            return None
        
        column_name = match_data.get("column_name")
        confidence = match_data.get("confidence_score", 0)
        reasoning = match_data.get("reasoning", "")
        field_understanding = match_data.get("field_understanding", "")
        context_analysis = match_data.get("context_analysis", "")
        
        # Ensure the selected column exists in available columns
        if column_name not in available_columns:
            #print(f" LLM selected invalid column '{column_name}' for '{bracketed_field}' - not in available columns", file=sys.stderr)
            return None
        
        # Check minimum confidence threshold
        if confidence < 70:
            print(f" LLM confidence too low ({confidence}%) for '{bracketed_field}' -> '{column_name}'", file=sys.stderr)
            return None
        
        #print(f" ✓ LLM intelligent match: '{bracketed_field}' -> '{column_name}' ({confidence}%)", file=sys.stderr)
        #print(f"   Field understanding: {field_understanding}", file=sys.stderr)
        #print(f"   Context analysis: {context_analysis}", file=sys.stderr)
        #print(f"   Reasoning: {reasoning}", file=sys.stderr)
        
        return FieldMatch(
            column_name=column_name,
            confidence_score=confidence,
            reasoning=reasoning,
            match_type="SEMANTIC_INTELLIGENT"
        )
        
    except Exception as e:
        print(f" Error in LLM field selection for '{bracketed_field}': {e}", file=sys.stderr)
        return None

# ---------------------------------------------------------------------------
# MAIN PROCESSING LOGIC
# ---------------------------------------------------------------------------
async def main():
    """Main processing function with mandatory environment ID validation"""
    global AGENT_ID, AGENT_SECRET_KEY, llm_provider, client, anthropic_client
    try:

        print("[STARTUP] Loading credentials from AWS Secrets...", file=sys.stderr)
        AGENT_ID, AGENT_SECRET_KEY = await fetch_credentials_from_api()

        print(f"[STARTUP]    Agent ID: \"{AGENT_ID}\"", file=sys.stderr)
        print(f"[STARTUP]    Secret Key: \"{AGENT_SECRET_KEY}\"", file=sys.stderr)
        
        selectional_memory = SimpleMemory()
        creational_memory = SimpleMemory()

        relationship_graph = {}
        memory_key_index = {}

        print(f" Connecting to MCP server: {MCP_URL}", file=sys.stderr)
        print(f" Agent ID: {AGENT_ID}", file=sys.stderr)
        
        # Connect to WebSocket
        async with ws_connect(MCP_URL) as ws:
            print("Connected to MCP server", file=sys.stderr)
            
            if not await authenticate_with_server(ws):
                    print(" Failed to authenticate with server", file=sys.stderr)
                    print_json_output([])
                    return
            print(" Authentication completed successfully", file=sys.stderr)
            # STEP 1: Read test cases FIRST - this is mandatory
            print("DEBUG: Reading test cases...", file=sys.stderr)
            await read_test_cases()

            project_id = None
            for tc in TEST_CASES:
                if "projectId" in tc and tc["projectId"]:
                    project_id = tc["projectId"]
                    break
                test_data = tc.get("testData", {})
                if isinstance(test_data, dict) and "projectId" in test_data and test_data["projectId"]:
                    project_id = test_data["projectId"]
                    break
            
            if not project_id:
                print("ERROR: No project ID found in any test case. Process cannot continue.", file=sys.stderr)
                raise ValueError("Project ID is mandatory but not found in test cases")
            
            provider_name = get_llm_provider_for_project(project_id)
            
            if provider_name == "OpenAI":
                OPENAI_API_KEY = get_openai_api_key_from_api()
                llm_provider = "openai"
                client = OpenAI(api_key=OPENAI_API_KEY)
                anthropic_client = None
            elif provider_name == "Anthropic":
                ANTHROPIC_API_KEY = get_anthropic_api_key_from_api()
                llm_provider = "anthropic"
                anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                client = None
            else:
                print(f"ERROR: Unknown provider name: {provider_name}", file=sys.stderr)
                raise RuntimeError(f"Unknown provider name: {provider_name}")
            
            if not TEST_CASES:
                print("ERROR: No test cases found. Process cannot continue.")
                raise ValueError("Test cases are required")
            
            print(f"DEBUG: Loaded {len(TEST_CASES)} test cases", file=sys.stderr)
            
            # STEP 2: Validate environment ID exists
            extracted_env_id = extract_environment_id_from_testcases(TEST_CASES)
            if extracted_env_id is None:
                print("ERROR: No environment ID found in any test case. Process cannot continue.", file=sys.stderr)
                print("ERROR: Environment ID must be provided in test case data.", file=sys.stderr)
                raise ValueError("Environment ID is mandatory but not found in test cases")
            
            print(f"DEBUG: Validated environment ID: {extracted_env_id}", file=sys.stderr)
            
            # STEP 3: Get database schema with the validated environment ID
            print("DEBUG: Getting database schema with validated environment ID...", file=sys.stderr)
            await get_db_schema(ws)
            
            if not DB_SCHEMA:
                print(f"ERROR: Failed to load database schema for environment {CURRENT_ENVIRONMENT_ID}", file=sys.stderr)
                raise ValueError(f"Database schema could not be loaded for environment {CURRENT_ENVIRONMENT_ID}")
            
            print(f"DEBUG: Successfully loaded schema with {len(DB_SCHEMA)} tables for environment {CURRENT_ENVIRONMENT_ID}", file=sys.stderr)
            
            relationship_graph = build_relationship_inference_layer(DB_SCHEMA)
            # Process each test case (rest of your existing logic remains the same)
            all_test_data = []
            
            for tc in TEST_CASES:
                tc_id = tc.get("tcSeqNo", tc.get("tcId", "unknown"))
                tc_description = tc.get("tcDescription", tc.get("title", "No description"))
                print(f"\nProcessing test case: {tc_id} - {tc_description}", file=sys.stderr)
                
                # Classify test case
                classification = await classify_tc(tc)
                
                # Get field analysis for consistency check
                field_analysis = await analyze_fields_for_tc(tc)
                selectional_count = len(field_analysis["selectional"])
                creational_count = len(field_analysis["creational"])
                
                # Consistency validation
                expected_classification = (
                    "selectional" if selectional_count > 0 and creational_count == 0 else
                    "creational" if creational_count > 0 and selectional_count == 0 else
                    "hybrid" if selectional_count > 0 and creational_count > 0 else
                    classification  # Keep original if no fields detected
                )
                
                if classification != expected_classification:
                    print(f"Classification inconsistency detected: {classification} vs expected {expected_classification}", file=sys.stderr)
                    print(f"Field counts - Selectional: {selectional_count}, Creational: {creational_count}", file=sys.stderr)
                    classification = expected_classification
                    print(f"Corrected classification to: {classification}", file=sys.stderr)
                
                # Process based on corrected classification and get ordered results
                result_data_list = []
                
                if classification == "selectional":
                    result_data_list = await process_selectional_testcase(ws, tc, selectional_memory, relationship_graph, memory_key_index)
                elif classification == "creational":
                    result_data_list = await process_creational_testcase(ws, tc, creational_memory)
                else:  # hybrid
                    result_data_list = await process_universal_hybrid(ws, tc, selectional_memory, creational_memory, relationship_graph, memory_key_index)
                
                # Validate results before adding to output
                if result_data_list:
                    # Create a temporary dict for validation
                    temp_dict = {}
                    for item in result_data_list:
                        if isinstance(item, dict) and "fieldName" in item and "value" in item:
                            temp_dict[item["fieldName"]] = item["value"]
                    
                    if await validate_generated_data(temp_dict, tc):
                        all_test_data.extend(result_data_list)
                        print(f"Test case {tc_id} processed successfully: {len(result_data_list)} fields", file=sys.stderr)
                    else:
                        print(f"Data validation failed for {tc_id}, excluding from output", file=sys.stderr)
                else:
                    print(f"No data generated for test case {tc_id}", file=sys.stderr)
                
                # Print memory contents for debugging
                print(f"\nMemory State After {tc_id}:", file=sys.stderr)
                print(f"  Selectional entities: {len(selectional_memory.get_all())}", file=sys.stderr)
                print(f"  Creational entities: {len(creational_memory.get_all())}", file=sys.stderr)
                print(f"  Memory key index: {list(memory_key_index.keys())}", file=sys.stderr)
            # Output final JSON result
            print_json_output(all_test_data)
    
    except Exception as e:
        print(f"FATAL ERROR: {e}", file=sys.stderr)
        print("Process terminated due to missing environment ID or other critical error.", file=sys.stderr)
        # Even on error, try to output valid JSON structure
        print_json_output([])
        sys.exit(1)
if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import json, json5
import logging
import os
import re
from typing import Any, Dict, List, Optional
import boto3
from urllib.parse import urlparse 
from websockets.client import connect as ws_connect
from dotenv import load_dotenv
from pydantic import BaseModel
from openai import AsyncOpenAI, OpenAI
from app import generate_or_update_pom
from app2 import process_ats
from pathlib import Path
import sys
import requests
import anthropic
sys.stdout.flush()
print(f"Anthropic version: {anthropic.__version__}")
# ── Authentication Variables ─────────────────────────────────────────────────
AGENT_ID: str = ""
AGENT_SECRET_KEY: str = ""
CURRENT_JWT_TOKEN: str = ""

llm_provider: str = ""
openai_client = None
anthropic_client = None

# ── Config & Logging ─────────────────────────────────────────────────────────
# 1) Locate the directory of this script (i.e. testDataMcp/)
base_dir = Path(__file__).resolve().parent

# 2) Load the .env in testDataMcp → brings in OPENAI_API_KEY
load_dotenv(dotenv_path=base_dir / ".env")

# 3) Load the root .env (one level up), but don't overwrite existing vars
load_dotenv(dotenv_path=base_dir.parent / ".env", override=False)

# 4) Now pull your vars
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")      # from testDataMcp/.env
MCP_URL = "ws://localhost:8931/"

#if not OPENAI_API_KEY:
#    raise RuntimeError("Please set OPENAI_API_KEY in your environment")

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def get_llm_provider_for_project(project_id: int) -> Optional[str]:
    """Fetch LLM provider name for the given project ID from the API."""
    url = "http://localhost:3000/api/getLlmProviderForProjectId"
    payload = {"projectId": project_id}
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()
        provider_name = data.get("data")
        print(f"\n[INFO] LLM Provider for project {project_id}: {provider_name}\n")
        return provider_name
    except Exception as e:
        print(f"[ERROR] Could not fetch LLM provider for project {project_id}: {e}")
        return None

# ── Authentication Functions ─────────────────────────────────────────────────
async def fetch_credentials_from_api() -> tuple[str, str]:
    """Fetch agent credentials from the API endpoint"""
    try:
        logger.info("[AUTH] Fetching credentials from API endpoints")
        
        # Fetch agent ID
        agent_id_response = requests.get(
            "http://localhost:3000/api/secrets/AGENT_ID",
            headers={'Content-Type': 'application/json'}
        )
        agent_id_response.raise_for_status()
        agent_id_data = agent_id_response.json()
        
        # Fetch secret key
        secret_key_response = requests.get(
            "http://localhost:3000/api/secrets/AGENT_SECRET_KEY", 
            headers={'Content-Type': 'application/json'}
        )
        secret_key_response.raise_for_status()
        secret_key_data = secret_key_response.json()
        
        agent_id = agent_id_data['value']
        secret_key = secret_key_data['value']
        
        logger.info(f"[AUTH] Agent ID: {agent_id}")
        logger.info(f"[AUTH] Secret Key: {secret_key[:20]}...")
        
        return agent_id, secret_key
        
    except requests.RequestException as e:
        logger.error(f"[AUTH] Failed to fetch credentials from API: {e}")
        raise
    except Exception as e:
        logger.error(f"[AUTH] Error processing credentials response: {e}")
        raise

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
        
        return openai_key
        
    except requests.RequestException as e:
        logger.error(f"[CONFIG] Failed to fetch OpenAI API key from API: {e}")
        raise
    except Exception as e:
        logger.error(f"[CONFIG] Error processing OpenAI API key response: {e}")
        raise

def get_aws_access_key_from_api() -> str:
    """Fetch AWS Access Key from the API endpoint"""
    try:
        response = requests.get(
            "http://localhost:3000/api/secrets/AWS_ACCESS_KEY_ID",
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        data = response.json()
        return data['value']
    except Exception as e:
        logger.error(f"[CONFIG] Failed to fetch AWS Access Key: {e}")
        raise

def get_aws_secret_key_from_api() -> str:
    """Fetch AWS Secret Key from the API endpoint"""
    try:
        response = requests.get(
            "http://localhost:3000/api/secrets/AWS_SECRET_ACCESS_KEY",
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        data = response.json()
        return data['value']
    except Exception as e:
        logger.error(f"[CONFIG] Failed to fetch AWS Secret Key: {e}")
        raise

def get_aws_region_from_api() -> str:
    """Fetch AWS Region from the API endpoint"""
    try:
        response = requests.get(
            "http://localhost:3000/api/secrets/AWS_REGION",
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        data = response.json()
        return data['value']
    except Exception as e:
        logger.error(f"[CONFIG] Failed to fetch AWS Region: {e}")
        raise

def get_aws_bucket_name_from_api() -> str:
    """Fetch AWS Bucket Name from the API endpoint"""
    try:
        response = requests.get(
            "http://localhost:3000/api/secrets/AWS_BUCKET_NAME",
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        data = response.json()
        return data['value']
    except Exception as e:
        logger.error(f"[CONFIG] Failed to fetch AWS Bucket Name: {e}")
        raise

def get_anthropic_api_key_from_api() -> str:
    """Fetch Anthropic API key from the API endpoint"""
    try:
        logger.info("[CONFIG] Fetching Anthropic API key from API endpoint")
        
        response = requests.get(
            "http://localhost:3000/api/secrets/ANTHROPIC_API_KEY",
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        data = response.json()
        
        anthropic_key = data['value']
        logger.info(f"[CONFIG] Anthropic API Key loaded (length: {len(anthropic_key)} chars)")
        logger.info(f"[CONFIG] Anthropic API Key: {anthropic_key}")
        
        return anthropic_key
        
    except Exception as e:
        logger.error(f"[CONFIG] Failed to fetch Anthropic API key: {e}")
        raise

# ADD THIS ENTIRE FUNCTION:
async def handle_websocket_message(ws, message_data):
    """Handle incoming WebSocket messages including token refresh"""
    global CURRENT_JWT_TOKEN
    
    try:
        if isinstance(message_data, str):
            message = json.loads(message_data)
        else:
            message = message_data
            
        if message.get("method") == "token_refresh":
            params = message.get("params", {})
            new_token = params.get("newToken")
            expires_in = params.get("expiresIn")
            refreshed_at = params.get("refreshedAt")
            
            if new_token:
                CURRENT_JWT_TOKEN = new_token
                logger.info(f"[JWT] Token refreshed automatically at {refreshed_at}")
                logger.info(f"[JWT] New token expires in: {expires_in}")
                logger.info(f"[JWT] Updated stored token (length: {len(CURRENT_JWT_TOKEN)} chars)")
                logger.info(f"[JWT] New JWT Token: {CURRENT_JWT_TOKEN}")
            return True
            
        return False
        
    except json.JSONDecodeError as e:
        logger.error(f"[JWT] Failed to parse WebSocket message: {e}")
        return False
    except Exception as e:
        logger.error(f"[JWT] Error handling WebSocket message: {e}")
        return False
    
# Load all configuration values (ADD THIS AFTER THE FUNCTION DEFINITIONS)
#OPENAI_API_KEY = get_openai_api_key_from_api()
#ANTHROPIC_API_KEY = get_anthropic_api_key_from_api() if USE_ANTHROPIC_CLAUDE else None

AWS_ACCESS_KEY_ID = get_aws_access_key_from_api()
print("AWS_ACCESS_KEY_ID: ", AWS_ACCESS_KEY_ID)

AWS_SECRET_ACCESS_KEY = get_aws_secret_key_from_api()
print("AWS_SECRET_ACCESS_KEY: ", AWS_SECRET_ACCESS_KEY)

AWS_REGION = get_aws_region_from_api()
print("AWS_REGION: ", AWS_REGION)

AWS_BUCKET_NAME = get_aws_bucket_name_from_api()
print("AWS_BUCKET_NAME: ", AWS_BUCKET_NAME)

async def authenticate_with_server(ws) -> bool:
    """Perform initial authentication with server using static credentials"""
    global CURRENT_JWT_TOKEN
    
    if not AGENT_ID or not AGENT_SECRET_KEY:
        logger.error("[AUTH] Missing agent credentials")
        return False
    
    logger.info(f"[AUTH] Authenticating agent: {AGENT_ID}")
    
    auth_request = {
        "jsonrpc": "2.0",
        "id": 999,  # Special ID for auth
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
                logger.error(f"[AUTH] Authentication timeout after {timeout} seconds")
                return False
                
            try:
                response_raw = await asyncio.wait_for(ws.recv(), timeout=5)
                response = json.loads(response_raw)
                
                if response.get("id") == 999:
                    if "error" in response:
                        error = response["error"]
                        logger.error(f"[AUTH] Authentication failed: {error.get('message', 'Unknown error')}")
                        return False
                    elif "result" in response:
                        result = response["result"]
                        if result.get("success"):
                            CURRENT_JWT_TOKEN = result.get("jwtToken", "")
                            logger.info("[AUTH] Authentication successful!")
                            logger.info(f"[AUTH] JWT token received (length: {len(CURRENT_JWT_TOKEN)} chars)")
                            logger.info(f"[AUTH] JWT token received: {CURRENT_JWT_TOKEN}")
                            return True
                        else:
                            logger.error(f"[AUTH] Authentication failed: {result}")
                            return False
                    break
                else:
                    continue
                    
            except asyncio.TimeoutError:
                continue
            except json.JSONDecodeError as e:
                logger.error(f"[AUTH] Failed to parse authentication response: {e}")
                return False
                
    except Exception as e:
        logger.error(f"[AUTH] Authentication error: {e}")
        return False
    
    return False

# ── OpenAI Client ─────────────────────────────────────────────────────────────
#openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
#anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY) if USE_ANTHROPIC_CLAUDE else None
# ── JSON-RPC Models ───────────────────────────────────────────────────────────
class JsonRpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id:      int
    method:  str
    params:  Any
    jwtToken: Optional[str] = None

class JsonRpcResponse(BaseModel):
    jsonrpc: str
    id:      int
    result:  Optional[Any]            = None
    error:   Optional[Dict[str, Any]] = None

# ── Helper: send RPC and wait for matching ID ────────────────────────────────
async def send_mcp_request(ws, method: str, params: Any, req_id: int) -> JsonRpcResponse:
    global CURRENT_JWT_TOKEN
    # Include JWT token for all requests except authentication
    jwt_token = None if method == "authenticate" else CURRENT_JWT_TOKEN
    payload = JsonRpcRequest(id=req_id, method=method, params=params, jwtToken=jwt_token).json()
    logger.debug(f"→ Sending MCP request #{req_id}: {method} {params}")
    await ws.send(payload)

    timeout = 10.0 if method == "authenticate" else 30.0

    try:   
        try:
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=timeout)

                if await handle_websocket_message(ws, msg):
                    continue

                data = json.loads(msg)
                if data.get("id") != req_id:
                    continue
                resp = JsonRpcResponse.model_validate(data)
                if resp.error and resp.error.get("code") == 4004:
                    error_data = resp.error.get("data", {})
                    if "newToken" in error_data:
                        new_token = error_data.get("newToken")
                        expires_in = error_data.get("expiresIn")
                        
                        logger.info(f"[JWT] Token expired - updating with new token from server")
                        logger.info(f"[JWT] New token expires in: {expires_in}")
                        logger.info(f"[JWT] New JWT Token: {new_token}")
                        
                        # Update stored token
                        CURRENT_JWT_TOKEN = new_token
                    
                        # Retry the request with new token
                        logger.info(f"[JWT] Retrying {method} request with refreshed token")
                        retry_payload = JsonRpcRequest(id=req_id, method=method, params=params, jwtToken=CURRENT_JWT_TOKEN).json()
                        await ws.send(retry_payload)
                        continue

                if resp.error:
                    raise RuntimeError(f"MCP tool error: {resp.error}")
                logger.debug(f"← Received MCP response #{req_id}: {resp.result}")
                return resp
        except asyncio.TimeoutError:
            raise RuntimeError(f"Request timeout ({timeout}s) for method: {method}")
    except Exception as e:
        # Check if it's a connection error
        if "ConnectionResetError" in str(e) or "ConnectionClosedError" in str(e):
            raise RuntimeError(f"Server connection lost during {method} request.")
        raise RuntimeError(f"Request failed for {method}: {e}") 

async def call_llm_unified(messages: list, temperature: float = 0.0) -> str:
    """Unified async LLM calling function that uses either OpenAI or Anthropic based on flag"""
    
    if llm_provider == "anthropic":
        logger.debug("[LLM] Using Anthropic Claude Sonnet 3.7 model")
        try:
            # Convert OpenAI format to Anthropic format
            if messages[0]["role"] == "system":
                system_msg = messages[0]["content"]
                user_msgs = messages[1:]
            else:
                system_msg = None
                user_msgs = messages
            
            response = await anthropic_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=1000,
                temperature=temperature,
                system=system_msg,
                messages=user_msgs
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"[LLM] Anthropic Claude failed: {e}")
            raise
    elif llm_provider == "openai":
        logger.debug("[LLM] Using OpenAI GPT-4o model")
        try:
            resp = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"[LLM] OpenAI failed: {e}")
            raise
    else:
        raise RuntimeError(f"Unknown LLM provider: {llm_provider}")





#AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
#print("AWS_ACCESS_KEY_ID: ", AWS_ACCESS_KEY_ID)
#AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
#print("AWS_ACCESS_KEY_ID: ", AWS_SECRET_ACCESS_KEY)
#
#AWS_REGION = os.getenv('AWS_REGION')
#print("AWS_REGION: ", AWS_REGION)
#
#AWS_BUCKET_NAME = os.getenv('AWS_BUCKET_NAME')
#print("AWS_BUCKET_NAME: ", AWS_BUCKET_NAME)



s3 = boto3.client(
's3',
aws_access_key_id=AWS_ACCESS_KEY_ID,
aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
region_name=AWS_REGION
)
print("---------------------Connected to S3 Client----------------------")

# ── LLM invocation ───────────────────────────────────────────────────────────
async def   invoke_llm_for_action(step: str, elements: List[Dict[str,Any]]) -> List[Dict[str,Any]]:

    SYSTEM_PROMPT = """
  You are an expert Playwright automation assistant.

  You will be given:
  - The entire test case
  - A specific test step (e.g. “Click on ‘Create Project’ button”)
  - The full page DOM info as a JSON array; each object has:
      • tagName, id, role, text, type, value 
      • selectors: an array of candidate Playwright locators 
      • options: for <select>, radio‐group or checkbox‐group, the list of labels 
  - The **expectedResult** string for that step (what you should verify)

  Your job:
  1. Identify the *one* element that the step refers to:
      - If the step says “Click … button”:
        • Filter the DOM array to the object with `tagName === "button"` and `text` exactly matching the step’s text. 
        • Ignore *all* other elements, even if their text is the same.
       For click actions, ONLY return selectors for clickable elements:
        • Button elements: locator('button'), getByRole('button')
        • Submit inputs: locator('input[type="submit"]')
        • Links: locator('a')
        • Text-based: getByText('ButtonText')
      - NEVER include form containers, divs, or non-interactive elements for click actions
      - For forms like #login-form, find the submit button INSIDE the form instead

      - Otherwise, apply your normal matching logic by tag/type/text.

  2. For that element, return *all* of its selectors (from its `selectors` array), ordered by priority:
      1. id (`locator('#myId')`)
      2. other attribute CSS (`locator('button[type="button"]')`)
      3. `page.getByLabel(...)`
      4. `page.getByPlaceholder(...)`
      5. `page.getByRole(...)` (with correct role and name)
      6. `page.getByText(...)`
      7. `nth-child(...)` fallback

  3. For select/radio/checkbox steps:
      • **Identify the target field** and the **exact text/value to be selected (`<TEXT_TO_SELECT>`)** from the `action` string.
      • **Determine the `selector` and `value` for `browser_select_option`:**
          - Find the single best Playwright selector for the main control element of the selection (e.g., the `<select>` tag itself, or the main container `div` for a custom dropdown, or the specific radio/checkbox input).
          - **To determine the `value` parameter for selection:**
            • **First, examine the identified element's `options` array (if present in the `elements` data). If there is an option where its `text` property matches `<TEXT_TO_SELECT>` AND it has a distinct `value` attribute, then emit that specific `value` attribute.**
            • **Otherwise (if no such distinct `value` attribute is found for the matching text within the element's `options`, or if the element does not expose an `options` array in that manner): Emit the `exact, full text <TEXT_TO_SELECT>` itself, as provided in the action description.**
            • If the matching option has a non‑empty value attribute, emit that value; otherwise emit the exact text (<TEXT_TO_SELECT>).
 4. For steps involving asserting specific values within structured **HTML tables** for a **single, uniquely identified row** (e.g."Assert that cell in 'Email' column for 'Project A' row has value 'projectA@example.com'", "Check the balance for account ABC"):
    • Use the `assertCellValueInRow` action.
    • Determine the `table_selector`. **Prefer robust and descriptive selectors like class names (`.table-class`) or semantic roles (`page.getByRole('table')`) over fragile positional selectors (e.g., `nth-child`).**
    • Identify the `row_identifier_column` (e.g.,"ID") and its `row_identifier_value` from the test step. This column should contain a unique value to find the specific row.
    • Identify the `target_column` (e.g."Balance") and its `expected_value` from the test step.

 5. For steps involving asserting that **all visible values in a specific HTML table column** match an expected value/pattern, or that **no values in a column match** a specified value/pattern (e.g. "Ensure no entries in 'Expiration Date' column are 'N/A'", "Confirm all 'Product Type' entries include 'Software'"):
    • Use the `assertTableColumnValues` action.
    • Determine the `table_selector`. **Prefer robust and descriptive selectors like class names (`.table-class`) or semantic roles (`page.getByRole('table')`) over fragile positional selectors (e.g., `nth-child`).**
    • Identify the `column_header` (e.g."Category") from the test step. This is the text of the column header whose values should be asserted.
    • Identify the `expected_value` from the test step. This is the value that *all* visible cells in the specified column should match (for positive assertions) or *none* should match (for negative assertions).
    • Determine the `match_type` (either `"exact"` or `"includes"`). Default to `"includes"` if not explicitly derivable from the step, but prefer `"exact"` if the step implies a precise match (e.g., "Product is 'Sold'").
    • **Crucially, determine the `assertion_type`. This dictates how `expected_value` is asserted across the column. Choose one of:**
         • `"all"`: When the step implies that **every** visible cell in the column must match the `expected_value` (e.g., "all entries are...", "every id is...", "confirm only X values are present"). This is the default.
         • `"none"`: When the step implies that **no** visible cell in the column should match the `expected_value` (e.g., "no entries are...", "should not contain...", "is not displayed").
         • `"any"`: When the step implies that **at least one** visible cell in the column must match the `expected_value` (e.g., "is visible", "is present", "contains X", "both X and Y are visible").
 
 6. • For verifying specific conditions on **filtered rows within a table** (e.g., "ensure no rejected applications from new users are present", "verify approved items have valid IDs"):
    • Use the `browser_assert_filtered_table_rows` action.
    • Provide `table_selector` (array of selectors).
    • Specify `filter_conditions` as an array: `[{ "column": "ColName", "value": "Val", "match_type": "exact"|"includes" }]`.
    • Set `assert_column`, `assert_value`, `assert_match_type`, and `assert_negative_assertion` (true if verifying absence).
 
 7. •For steps involving verication of text visible use assertText tool.

 Strictly Follow this output format:
 ```json
  [
    { // For standard UI interactions
      "action": "click"|"fill"|"select"|"selectOption"|"check"|"uncheck"| "navigate"|"assertText"|"assertUrlContains"|"assertValue"|
                "assertSelectedOption"|"assertElementVisible"|"waitForSelector"|"assertCellValueInRow",
      "selector": [ /* your ordered page.locator(...) calls */ ],
      "text": "...",         // only for fill/assertText
      "value": "...",        // only for selectOption
      "url": "...",           // only for navigate
      "table_selector": ["...", "..."], *only for assertCellValueInRow and assertTableColumnValues* // Array of Playwright selectors for the table
      "row_identifier_column": "...",    *only for assertCellValueInRow*// Text of column header (e.g., "Name")
      "row_identifier_value": "...",    *only for assertCellValueInRow*// Value in the identifier column (e.g., "Robert Johnson")
      "target_column": "...",           *only for assertCellValueInRow*// Text of column header to assert (e.g., "Status")
      "expected_value": "...",           *only for assertCellValueInRow*// Expected value in the target cell (e.g., "Approved")
      "column_header": "...",          *only for assertTableColumnValues*  // Text of column header to assert (e.g., "Status")
      "expected_value": "...",          *only for assertTableColumnValues*// Expected value for all cells (e.g., "Approved")
      "match_type": "exact"|"includes",  *only for assertTableColumnValues*// How to match (e.g., "exact", "includes")
      "assertion_type": ..., //  *only for assertTableColumnValues* 
      "filter_conditions": [{"column": "...", // Header text or 1-based index of the column for this filter.
                "value": "...", // Expected value/pattern for the filter column.
            "match_type": "exact"|"includes" // Comparison method for this filter.}],
      "assert_column": "...", // Header text or 1-based index of the column to assert values in.
      "assert_value": "...", // The value/pattern to assert against in the assert_column.
      "assert_match_type": "exact"|"includes", // Comparison method for the final assertion.
      "assert_negative_assertion": true|false // Set to true if asserting the absence of assert_value.      
    },
    
  ]
  Return only the JSON—no commentary, no markdown fences.
  """
    #user_payload = {"step": step,"elements": elements}
    #resp = await openai_client.chat.completions.create(
    #    model="gpt-4o",
    #    messages=[
    #        {"role": "system", "content": SYSTEM_PROMPT},
    #        
    #        {"role": "user",   "content": json.dumps(user_payload, indent=2)}
    #    ],
    #    temperature=0.0,
    #)
    #text = resp.choices[0].message.content
    user_payload = {"step": step,"elements": elements}
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload, indent=2)}
    ]
    
    text = await call_llm_unified(messages, temperature=0.0)
    print("TEXT: ",text)
    logger.debug(f"← Raw LLM response:\n{text!r}")
    json_match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
    if json_match:
        json_content = json_match.group(1).strip()
    else:
        # Fallback if markdown fences are not present (though prompt asks for them)
        # Attempt to clean up common LLM preamble/postamble before parsing
        json_content = text.strip()
        if json_content.startswith("```"): # Remove generic ``` if no `json` after it
            json_content = json_content[3:]
        if json_content.endswith("```"):
            json_content = json_content[:-3]
        json_content = json_content.strip()
    try:
        obj = json.loads(json_content)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON content: {e}. Content attempting to parse:\n{json_content!r}")
        # If parsing fails here, it means the content is truly malformed JSON.
        # It's better to return an empty list or raise an error rather than proceed with bad data.
        return [] # Or raise some specific error if you prefer
    
    if isinstance(obj, list):
        # Filter out any non-dictionary items that might sneak in
        actions = [item for item in obj if isinstance(item, dict)]
        if len(actions) != len(obj):
            logger.warning(f"LLM returned a list with non-dictionary items. Filtered to: {actions}")
    elif isinstance(obj, dict):
        actions = [obj] # Wrap single dictionary in a list
    else:
        logger.error(f"LLM returned unexpected JSON root type: {type(obj)}. Expected list or dict. Skipping actions.")
        actions = [] # Treat as no valid actions if the root is not a list or dict

    print("ACTIONS (after robust parsing): ",actions)

    # try:
    #     obj = json.loads(text)
    # except Exception:
    #     start, end = text.find("["), text.rfind("]") + 1
    #     obj = json.loads(text[start:end])
    # actions = obj if isinstance(obj, list) else ([obj] if isinstance(obj, dict) else [])
    # print("ACTIONS: ",actions)
    # # ── Carry the quoted button text out of the step and into act["text"] ──
    if actions:
        act0 = actions[0]
        if act0.get("action") == "click" and "button" in step.lower() and "text" not in act0:
            m = re.search(r"'([^']+)'", step)
            if m:
                act0["text"] = m.group(1)

    # ── filter out non-<button> locators on click-the-button steps ─────────
    # if actions:
    #     act0 = actions[0]
    #     if act0.get("action") == "click" and "button" in step.lower():
    #         raw = act0.get("selector", [])
    #         filtered = [
    #             s for s in raw
    #             if s.startswith("locator('button") or "getByRole('button" in s
    #         ]
    #         if filtered:
    #             logger.debug(f"Filtering click selectors down to buttons: {filtered}")
    #             act0["selector"] = filtered

    return actions


async def invoke_llm_for_verification(
    expectedResult: str,
    used_locators: List[str],
    elements: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Given:
      - expectedResult: the string you want to verify
      - used_locators: the list of Playwright locator strings you just used for the action
      - elements: the fresh DOM info after the action
    Returns exactly one JSON object describing the assertion to run.
    """
    SYSTEM_PROMPT = """
Your job is to generate a JSON **array** of one or more JSON objects, where each object defines a verification step based on the provided `expectedResult` and available `elements`.

The goal is to select the most robust and semantically appropriate Playwright selector for each verification.
### Guidance on `used_locators`
You also receive **`used_locators`**, the exact Playwright selector strings that were just employed to perform the preceding action.  
• Treat every `used_locator` as a **hint only**.  
• **If** a `used_locator` cleanly and robustly identifies the element that should be verified, include that locator (along with any other stronger selectors you discover for the same element) in the returned `selector` array, maintaining the usual robustness‑first ordering.  
• **If** none of the `used_locators` are appropriate—because they are fragile, unrelated, or point to the wrong element—**ignore them completely** and derive selectors solely from the fresh DOM `elements` snapshot, following the rules below.  
• Never return a `used_locator` that doesn’t actually correspond to the element being verified.

- **For an object verifying a field’s value (e.g., after typing into an input)**:
    • **TRIGGER:** If the `expectedResult` indicates a field being 'filled', 'populated', 'entered', or similar terms for text input, use action `"assertValue"`.
    • **Extract the precise field name (e.g., 'Username', 'password field') and the exact expected value from the `expectedResult` string.**
    • Scan the `elements` array. Find the single best `input` or `textarea` element whose `text` property (from its associated label), `placeholder`, or `id` strongly indicates it is the target field for the extracted field name.
    • **Return an array of ALL relevant and stable selectors from the chosen element's `selectors` list, ordered by robustness (most robust first).** Prioritize:
        1.  Any `page.locator('#<id_of_field>')` if the element has a unique and descriptive ID.
        2.  Any `page.locator('.form-group:has(label:has-text("<extracted field name>")) input')` if it exists in the element's selectors and matches the field name.
        3.  Any `page.getByLabel('<extracted field name>')` if available and matching.
        4.  Any `page.getByPlaceholder('<extracted field name>')` if available and matching.
        5.  Any other `page.locator()` style selector from the `selectors` list.
        6.  Any other `page.getBy...()` style selector from the `selectors` list.
    • **IMPORTANT FORMATTING NOTE: The "selector" property for assertValue MUST always be a JSON array, even if it contains only one selector.**
    • Format:
      ```json
      {
        "action":"assertValue",
        "selector":["<robust selector 1>", "<robust selector 2>", ...], // THIS IS NOW AN ARRAY
        "expected":"<extracted precise expected value from expectedResult>" // LLM: Extract the value here
      }
      ```

- **For an object verifying a selected option in a dropdown**:
    • **TRIGGER:** If the `expectedResult` indicates an option being 'selected', 'chosen', 'set', or similar terms for a dropdown, use action `"assertSelectedOption"`.
    • Identify the field name related to the selection (e.g., 'Domain', 'Automation Framework') and **extract the exact expected selected value from the `expectedResult` string.**
    • Find the `select` element or the main interactive element of the custom dropdown that was likely manipulated or is expected to have this option selected. Prioritize by `id`, then `name`, then `role="combobox"` or other specific locators for custom dropdowns.
    • Format:
      ```json
      {
        "action":"assertSelectedOption",
        "selector":"<chosen precise selector for the select element>", // This should be a single string
        "expected":"<extracted precise expected value from expectedResult>" // LLM: Extract the value here
      }
      ```
    - **If it is NOT a native `<select>` tag (e.g., a custom dropdown like React-Select, typically composed of `div`, `span`, or `input` elements) AND the `expectedResult` implies verifying the *displayed text of the selected option* (e.g., 'Text X is visible for the dropdown'), use action `"assertDisplayedOptionText"`.**
    • **Extract the exact expected displayed text value from the `expectedResult` string.**
    • **Identify the element that *displays* the selected text for this custom dropdown from the `elements` array.**

    • **PRIORITY AND SELECTION FOR SELECTORS (FROM THE CHOSEN ELEMENT'S `selectors` LIST):**
        • **Always aim to return an array containing ALL robust and stable selectors found for the *chosen element* that visually represents the text. Prioritize them by robustness.**
        1.  **`page.getByRole('option', { name: '...' })`**: If the element has `role="option"` and `name` matches the text, this is highly robust.
        2.  **`page.locator('#<id>')`**: If the element has a unique and descriptive `id`, include this.
        3.  **`page.getByText('<exact_text>')`**: Include if it directly matches the text of the *chosen element*.
        4.  Any other `page.locator()` style selector from the element's `selectors` list (e.g., `page.locator('span:nth-child(XYZ)')`).
        5.  Any other `page.getBy...()` style selector from the element's `selectors` list.
        • **If multiple selectors exist for the *same* intended element (e.g., `page.getByText('KRI')`, `page.getByRole('option', { name: 'KRI' })`, and `page.locator('#react-select-6--value-item')` all point to the KRI option), include ALL of them in the `selector` array.**
    • **IMPORTANT FORMATTING NOTE: The "selector" property for assertDisplayedOptionText MUST always be a JSON array, even if it contains only one selector.**
    **Use assertDisplayedOptionText only for verifying any particular values not just common options**
    • Format:
      ```json
      {
        "action":"assertDisplayedOptionText",
        "selector":["<robust selector 1>", "<robust selector 2>", ...], // ENSURE THIS IS ALWAYS AN ARRAY with ALL relevant selectors
        "expected":"<extracted precise expected displayed text value from expectedResult>" // LLM: Extract the value here
      }
      ```
    

- **For an object verifying browser navigation or redirect**:
    • Use action `"assertUrlContains"`.
    • Provide `"fragment"` as the key page name or URL fragment (no quotes).
    • Ignore `used_locators`.
    • Format:
      ```json
      {
        "action":"assertUrlContains",
        "fragment":"<page name or URL fragment>"
      }
      ```

- **For an object verifying the visibility of a specific HTML element (e.g., an input field, button, image, div)**:
    • Use action `"assertElementVisible"`.
    • Use this to verify elements or existing of elements other than expected result containing any specific values to verify
    • **Identify the element** mentioned in the `expectedResult` (e.g., "Email input field", "Login button", "logo image").
    • Scan the `elements` array. Find the single best element that matches the description.
    • **Prioritize choosing a selector for the chosen element based on these rules, in order:**
        1.  **`page.locator('#<id_of_element>')`**: If the element has a unique and descriptive ID.
        2.  **`page.getByLabel('<label_text>')`**: If the element is an input, textarea, or select, and has a clear associated label. Use this to target the *input element itself*.
        3.  **`page.getByRole('<role>', { name: '<accessible_name>' })`**: If the element has a clear role and accessible name (e.g., `page.getByRole('button', { name: 'Login' })`).
        4.  **`page.getByPlaceholder('<placeholder_text>')`**: If the element is an input or textarea and has a matching placeholder.
        5.  Any other precise and stable selector from the `selectors` list of the chosen element.
    • Format:
      ```json
      {
        "action":"assertElementVisible",
        "selector":"<chosen precise selector for the element>" // This must be a single string
      }
      ```

- **For any other generic text assertion (e.g., titles, messages, generic paragraphs, text within a label element)**:
    • Use action `"assertText"`.
    • **This action should be used for verifying the visible *text content* on the page, not for checking the visibility of interactive elements like input fields.**
    • **Extract the precise text to verify from the `expectedResult` string.**
    • Scan the `elements` array. Find the single best element that visually represents this extracted text on the page.
    • **Prioritize choosing a selector for the chosen element based on these rules, in order:**
        1.  **`page.getByText('<exact_extracted_text>')`**: This is the **primary and most recommended choice** for verifying the visual presence of text. Use this when the extracted text directly appears as the `text` content of any element (like a `<span>`, `<div>`, `<a>`, `<h1>`, or specifically a `<label>` element that displays the text).
        2.  **`page.getByRole('<role>', { name: '<extracted text>' })`**: If the extracted text is the accessible name of an element with a specific role (e.g., a button with "Login" text).
        3.  Any other precise and stable selector (like `page.locator('#id')`, `page.locator('.class')`) from the `selectors` list of the chosen element, if `getByText` or `getByRole` are not suitable or available.
    • **IMPORTANT FORMATTING NOTE: The "selector" property for assertText MUST always be a JSON array, even if it contains only one selector.**
    • Format:
      ```json
      {
        "action":"assertText",
        "selector":["<chosen selector for the element that best represents the text>"], // ENSURE THIS IS ALWAYS AN ARRAY
        "text":"<the extracted precise text to verify>"
      }
      ```

- **For steps involving asserting specific values within structured **HTML tables** for a **single, uniquely identified row** (e.g., "Verify status for Robert Johnson is Approved", "Assert that cell in 'Email' column for 'Project A' row has value 'projectA@example.com'", "Check the balance for account ABC"):
    • Use the `assertCellValueInRow` action.
    • Determine the `table_selector`. **Prefer robust and descriptive selectors like class names (`.table-class`) or semantic roles (`page.getByRole('table')`) over fragile positional selectors (e.g., `nth-child`).**
    • Identify the `row_identifier_column` (e.g., "Name", "ID") and its `row_identifier_value` from the test step. This column should contain a unique value to find the specific row.
    • Identify the `target_column` (e.g., "Status", "Balance") and its `expected_value` from the test step.
    •format:
    ```json
      {
        "action": "assertCellValueInRow",
        "table_selector": ["<selector 1>", "<selector 2>", ...],
        "row_identifier_column": "<header text or column key used to locate the correct row>",
        "row_identifier_value": "<value that identifies the target row>",
        "target_column": "<header text or column key for the cell to check>",
        "expected_value": "<exact expected cell value>"
      }
      ```
- **For steps involving asserting that **all visible values in a specific HTML table column** match an expected value/pattern, or that **no values in a column match** a specified value/pattern (e.g. "Ensure no entries in 'Expiration Date' column are 'N/A'", "Confirm all 'Product Type' entries include 'Software'"):
    • Use the `assertTableColumnValues` action.
    • Determine the `table_selector`. **Prefer robust and descriptive selectors like class names (`.table-class`) or semantic roles (`page.getByRole('table')`) over fragile positional selectors (e.g., `nth-child`).**
    • Identify the `column_header` (e.g."Category") from the test step. This is the text of the column header whose values should be asserted.
    • Identify the `expected_value` from the test step. This is the value that *all* visible cells in the specified column should match (for positive assertions) or *none* should match (for negative assertions).
    • Determine the `match_type` (either `"exact"` or `"includes"`). Default to `"includes"` if not explicitly derivable from the step, but prefer `"exact"` if the step implies a precise match (e.g., "status is 'Approved'").
    • **Crucially, determine the `assertion_type`. This dictates how `expected_value` is asserted across the column. Choose one of:**
         • `"all"`: When the step implies that **every** visible cell in the column must match the `expected_value` (e.g., "all entries are...", "every id is...", "confirm only X values are present"). This is the default.
         • `"none"`: When the step implies that **no** visible cell in the column should match the `expected_value` (e.g., "no entries are...", "should not contain...", "is not displayed").
         • `"any"`: When the step implies that **at least one** visible cell in the column must match the `expected_value` (e.g., "is visible", "is present", "contains X", "both X and Y are visible").
 
  •format:
    ```json
      {
        "action": "assertTableColumnValues",
          "table_selector": ["...", "..."], // Array of Playwright selectors for the table
          "column_header": "...",           // Text of column header to assert 
          "expected_value": "...",          // Value to match/not match
          "match_type": "exact"|"includes", // How to match (e.g., "exact", "includes")
          "assertion_type": ... // Set to true if verifying absence of expected_value
      }
     ```
"""

#     




    # Build the user payload
    user_payload = {
        "expectedResult": expectedResult,
        "usedLocators": used_locators,
        "elements": elements
    }

    # Call the LLM
    #resp = await openai_client.chat.completions.create(
    #    model="gpt-4o",
    #    messages=[
    #        {"role": "system", "content": SYSTEM_PROMPT},
    #        {"role": "user",   "content": json.dumps(user_payload)}
    #    ],
    #    temperature=0.0
    #)
    #text = resp.choices[0].message.content
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": json.dumps(user_payload)}
    ]
    
    text = await call_llm_unified(messages, temperature=0.0)
    logger.debug(f"← Raw LLM response For Verification:\n{text!r}")
    match = re.search(r'```json\n(.*)\n```', text, re.DOTALL)
    if match:
        json_str = match.group(1).strip()
    else:
        # Fallback: if markdown not found, assume the whole response is JSON
        json_str = text.strip()

    # Now, parse the extracted JSON string
    try:
        # This will now correctly parse a JSON array
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from LLM: {e}")
        logger.error(f"Problematic JSON string: {json_str!r}")
        raise #


def normalize_selector(selector_string: str) -> str:
    """
    Normalizes Playwright selector strings to a consistent format.
    Removes 'page.' prefix, strips quotes and parentheses if they wrap the whole string.
    This aims to make selectors comparable regardless of their origin (MCP vs. LLM output).
    """
    # Remove 'page.' prefix if present
    normalized = selector_string.replace("page.", "")
    
    # Remove outer 'locator(', 'getByPlaceholder(', etc. and closing ')'
    # This is a bit more robust than just .strip("'\"()")
    match = re.match(r'^(locator|getByPlaceholder|getByText|getByRole|getByLabel)\((.*)\)$', normalized)
    if match:
        normalized = match.group(2) # Extract content inside the parentheses
    
    # Strip any remaining surrounding quotes
    normalized = normalized.strip("'\"")

    return normalized

# ── MAIN: connect → iterate steps → build test ──────────────────────────────
async def main():
    global AGENT_ID, AGENT_SECRET_KEY, CURRENT_JWT_TOKEN, llm_provider, openai_client, anthropic_client

    try:
        logger.info("[STARTUP] Loading credentials from AWS Secrets...")
        AGENT_ID, AGENT_SECRET_KEY = await fetch_credentials_from_api()
        logger.info("[STARTUP] Credentials loaded successfully")
    except Exception as e:
        logger.error(f"[STARTUP] Failed to load credentials: {e}")
        return  # Exit if we can't get credentials


    def sanitize(s: str) -> str:
        # 1) Quote ISO timestamps
        s = re.sub(r'(:\s*)(\d{4}-\d{2}-\d{2}T[\d:.]+Z)(\s*[},])', r'\1"\2"\3', s)

        # 2) 'NULL' -> null
        s = re.sub(r":\s*'NULL'", r": null", s)

        # 3) For certain fields, replace the whole RHS with a safely double‑quoted value.
        # We do this line-by-line so we don't get tripped by nested quotes.
        fields = [
            "tcStepDescription",
            "tcStepExpectedResult",
            "testCaseNumber",
            "tcPriority",
            "smoke",
            "regression",
            "uat",
        ]
        # Build a regex that captures the entire RHS up to end of line or trailing comma.
        pat = re.compile(rf'^(\s*(?:{"|".join(fields)})\s*:\s*)(.+?)(\s*,?\s*)$', re.M)

        def repl(m):
            prefix, rhs, suffix = m.groups()
            rhs = rhs.strip()
            # strip one layer of wrapping quotes if present
            if (rhs.startswith("'") and rhs.endswith("'")) or (rhs.startswith('"') and rhs.endswith('"')):
                rhs = rhs[1:-1]
            # escape backslashes and double quotes for JSON
            rhs = rhs.replace('\\', '\\\\').replace('"', '\\"')
            return f'{prefix}"{rhs}"{suffix}'

        s = pat.sub(repl, s)
        return s

#     api_response = """{
#   testCaseId: 12470,
#   projectId:424,
#   testCaseNumber: 'TC-59082-001',
#   tcDescription: "Verify that an administrator can navigate to Optional Field Configuration and toggle 'Show US Tax Exemptions menu' to ON",
#   testCasePrecondition: "User has Administrator role and valid credentials. 'Show US Tax Exemptions menu' is currently set to OFF.",
#   bddTestCase: 'NULL',
#   testData: { TestData:[
#     {
#       "fieldName": "USERNAME",
#       "value": "bob"
#     },
#     {
#       "fieldName": "PASSWORD",
#       "value": "test@123"
#     },   
     
#   ] },   
#   refTSId: 59082,
#   tcPriority: 'high',
#   tcCreatedDateTime: 2025-07-27T15:07:48.000Z,      
#   tcModifiedDateTime: null,
#   smoke: 'Y',
#   regression: 'Y',
#   uat: 'Y',
#   refPageObjectModelPath: 'ATS/NewProject/src/test/java/com/page/base/MenuPage.java',
#   testCasePath: 'ATS/NewProject/src/test/java/com/cognitest/testcases/TC_referenceTestCase.java',       
#   destinationPath: '321/Java + Playwright/apikeycheckinlocalthird/src/test/java/com/cognitest/testcases',
#   pageHook: 'ATS/NewProject/src/test/java/com/page/base/CognitestPageHook.java',
#   pageObjectModelPath: 'ATS/NewProject/src/test/java/com/page/base/MenuPage.java',
#   playwrightWrapper: 'ATS/NewProject/src/main/java/com/ui/base/PlaywrightWrapper.java',
#   destinationPathForPageObjectModelPath: '321/Java + Playwright/apikeycheckinlocalthird/src/test/java/com/page/base/MenuPage.java',
#   locatorsPath: '321/locators/',
#   steps: [
#     {
#       tcStepId: 71526,
#       tcStepNumber: 'TCS-12486-001',
#       tcStepDescription: 'navigate to the url "http://3.82.99.137:5001/login.html"',
#       tcStepExpectedResult: 'User is successfully navigated to the login page.'
#     },
#     {
#       tcStepId: 71527,
#       tcStepNumber: 'TCS-12486-002',
#       tcStepDescription: 'Enter '<<USERNAME>>' in the 'Username' field,
#       tcStepExpectedResult: ''Username' field is filled with '<<USERNAME>>''
#     },
#     {
#       tcStepId: 71528,
#       tcStepNumber: 'TCS-12486-003',
#       tcStepDescription: "Enter '<<PASSWORD>>' in the 'Password' field",
#       tcStepExpectedResult: 'Password field is filled with '<<PASSWORD>>''
#     },
#     {
#       tcStepId: 71529,
#       tcStepNumber: 'TCS-12486-004',
#       tcStepDescription: "Click 'Login' button",
#       tcStepExpectedResult: "'Application Cases' text must be visible"
#     },
#     {
#       tcStepId: 71530,
#       tcStepNumber: 'TCS-12486-005',
#       tcStepDescription: "Select 'Approved' from the 'All Statuses' dropdown",
#       tcStepExpectedResult: ''All Statuses' is selected successfully as 'Approved''
#     },
#     {
#       tcStepId: 71530,
#       tcStepNumber: 'TCS-12486-005',
#       tcStepDescription: "Verify that only approved applications are displayed",
#       tcStepExpectedResult: 'Only applications with status 'Approved' are visible'
#     }
    
#   ]
# }
# """
    api_response = sys.argv[1]
    print("API RESPONSE: ", api_response)
    clean = sanitize(api_response)
    
    clean = re.sub(
    r'(:\s*)(\d{4}-\d{2}-\d{2}T[\d:.]+Z)(\s*[},])',
    r'\1"\2"\3',
    clean,)


    clean = re.sub(r":\s*'NULL'", r": null", clean)

    
    data = json5.loads(clean)
    print("Data: ",data)
    

    print("json Response: ",data )



    steps_src = data.get("steps", []) or []
    test_case = {
        "testCaseId": data.get("testCaseId"),
        "testCaseNumber": data.get("testCaseNumber"),
        "tcDescription": data.get("tcDescription"),
        "testCasePrecondition": data.get("testCasePrecondition"),
        "tcPriority": data.get("tcPriority"),
        "steps": [
            {
                "tcStepId": s.get("tcStepId"),
                "tcStepNumber": s.get("tcStepNumber"),
                "action": s.get("tcStepDescription"),
                "expectedResult": s.get("tcStepExpectedResult"),
            }
            for s in steps_src
        ],
    }
    test_case_number = data.get("testCaseNumber")
    td_obj = data.get("testData") or {}
    td_list = td_obj.get("TestData") or []
    test_data_dict = { "TestData": td_list }

    test_data = json.dumps(test_data_dict, ensure_ascii=False).encode("utf-8")
    proj_id = data.get("projectId")
    input_pom_path = f"{proj_id}/ats/MCP_ATS/Playwright_Java/src/test/java/com/page/base/MenuPage.java"
    provider_name = get_llm_provider_for_project(proj_id)

    if provider_name == "OpenAI":
        OPENAI_API_KEY = get_openai_api_key_from_api()
        llm_provider = "openai"
        openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        anthropic_client = None
    elif provider_name == "Anthropic":
        ANTHROPIC_API_KEY = get_anthropic_api_key_from_api()
        llm_provider = "anthropic"
        anthropic_client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
        openai_client = None
    else:
        raise RuntimeError(f"Unknown provider name: {provider_name}")

    print("Test-Case loaded")
    print("Test-Case: ",test_case)
    print("Test-Data loaded")
    print("Test-Data: ",test_data)
    

    # with open("Underwriting_Test_Cases/UWtest_case1.json", 'r', encoding='utf-8') as f:
    #     test_case = json.load(f)
    # print("Loaded test case")

    # with open("Underwriting_Test_Cases/UWtest_data1.json", 'r', encoding='utf-8') as f:
    #     test_data = json.load(f)
    # print("Loaded test data")
    

    system_prompt = """
You are an expert JSON transformation assistant specialized in test case automation.
Your primary function is to combine a generic `test_case` JSON template with its corresponding `test_data` JSON.
Your goal is to produce a single, merged JSON object representing a concrete, executable test case.

**YOU MUST ADHERE TO THE FOLLOWING RULES ABSOLUTELY:**

1.  **OUTPUT FORMAT:**
    * Your response MUST be **ONLY a single, valid JSON object**.
    * DO NOT include any conversational text, explanations, or markdown code blocks. Just the raw JSON object.
    * All JSON keys must be unique. DO NOT duplicate any keys at any level.
    * Ensure proper JSON syntax: NO trailing commas, NO empty objects `{}`, NO extraneous brackets or braces `[] {}`.

2.  **TOP-LEVEL STRUCTURE & PRESERVATION:**
    * Retain ALL original top-level keys (`testCaseId`, `testCaseNumber`, `tcDescription`, `testCasePrecondition`,`tcPriority`, `steps`) from the `test_case` JSON and their original values.
    * **Preserve Top-Level `expectedResult`:** If a top-level `expectedResult` key exists, include it with its original value.

3.  **`steps` ARRAY TRANSFORMATION:**
    * The output `steps` array **MUST contain the EXACT SAME NUMBER of step objects** as in the input.
    * For each step object:
        1.  **Preserve `stepNumber`.**
        2.  **Transform `action` Fields:**
            * **Iterate and Inject Data:** For each `step` object, and for each `fieldName` and its corresponding `value` available in the `test_data` for the current test case:
            * **A. Specific Input/Selection Verbs (Patterned Rephrasing):** If the `action` explicitly uses verbs like "Enter", "Fill", "Select", or "Set" in conjunction with a `fieldName` (e.g., "Enter the Role", "Fill the Email", "Select the dropdown"), apply the following precise rephrasing and **prioritize this rule**:
                * **For "Enter" actions:** 
                 * **Single Field:** If the `action` uses "Enter" and corresponds to a single `fieldName` from `test_data`, **always rewrite it as:** "Enter '[VALUE]' in the '[FIELD_NAME]'." 
                 * **Multiple Fields:** For actions involving multiple fields (e.g., "Enter Username and Password"), format as: "Enter '[USERNAME_VALUE]' in Username and '[PASSWORD_VALUE]' in Password."
                * **For "Fill" actions:** Rewrite as "Fill '[FIELD_NAME]' with '[VALUE]'."
                * **For "Select" actions:
                    **Identify the `fieldName` (from `test_data`) that the original action is referring to.** This `fieldName` string must be used exactly as provided, including any spaces or special characters it may contain.
                    * **Rewrite the action as:** "Select '[FIELD_NAME]' as '[VALUE]'."
                    * **Example:** If `action` is "Select Department" and `fieldName` is "Department" with `value` "IT", then output: "Select 'Department' as 'IT'." (This applies whether 'Department' is a single word or if `fieldName` were, for instance, 'Risk Level'.)
                * **For "Set" actions:** Rewrite as "Set '[FIELD_NAME]' to '[VALUE]'."
            * **A1. Record-Targeting UI Actions (Generic Click/Open/View/Actions/etc.):**
                * When an `action` performs a UI operation (e.g., "Click", "Open", "View", "Actions", "Edit", "Delete", "Expand", "Go to") **on or for a specific record/row/item that mentions a `fieldName` token** (quoted or unquoted), you MUST inject the matching `value` from `test_data` even though the verb is not Enter/Fill/Select/Set.
                * **Minimal substitution is preferred:** Replace just the occurrence of the `fieldName` token with its `value`, preserving surrounding UI label text and quotes if present.
                  * Example pattern (generic): Input action: `Click on 'Actions' for '<FIELD_NAME>'` -> Output action: `Click on 'Actions' for '<VALUE>'`.
                * You may lightly rephrase for clarity (e.g., "...for the record '<VALUE>'") as long as you preserve the original UI intent and key literal tokens (buttons, menu labels, etc.).
            * **B. Generic Field Name Replacement (String Matching & Fallback):**
                * After applying rules **A** and **A1**, search the `action` string (case-insensitive) for each `fieldName` token from `test_data`, with or without surrounding single quotes.
                * If found **anywhere** in the string (including in phrases like "of the respective", "associated", "for", "corresponding", "selected", etc.), replace that occurrence with the corresponding `value`. This applies **regardless of the verb** (Click, Navigate, etc.) and guarantees data injection whenever a `fieldName` is mentioned.
            * **C. Implicit‑Value Verification (Generic)**
                * While processing the steps actions, keep track of the **most recent `{fieldName, value}` pair that you injected**.
                * If a later step’s `action` meets **all** of the conditions below,enrich it with that **value**:
                    1.  The `action` starts with the verb **“Verify”** or **“Assert”** (case‑insensitive).
                    2.  It contains **none** of the explicit `fieldName` tokens from `test_data`.
                    3.  It contains a word such as **“match”**, **“matches”**, **“matching”**, **“display(s)”**, **“filtered”**, or **“criteria”** — indicating that a prior input is being checked.
                * When these conditions are met:
                    * **Inject** the value by rewriting the phrase **after** the verb “match(es)”/“display(s)”/etc. so it reads `"… match(es) the value '<VALUE>'"`.
                    * Update the step’s `expectedResult` to repeat the same value, e.g. `"The displayed records match the value '<VALUE>'."`
                * If no prior `{fieldName, value}` pair exists, or the conditions above aren’t met, leave the step untouched.
            * **No Data Modification:** If a step's `action` does not contain any `fieldName` from `test_data` (after checking rules A, A1, and B), leave the `action` exactly as is.
        3.  **Transform `expectedResult` Fields:**
            * **For *every* step where you inject a `value` into the `action`, you *must* also rewrite its `expectedResult` to explicitly include that same `value`.**
            * **A. Single Value Input:**
                * **Entered/Filled:** "'[FIELD_NAME]' field is filled with the value '[VALUE]'."
                * **Selected:** "The '[FIELD_NAME]' is selected as '[VALUE]'."
            * **B. Multiple Value Input (e.g., Username and Password entered together):**
                * "The '[FIELD_NAME1]' field contains the entered '[VALUE1]' and the '[FIELD_NAME2]' field contains the entered '[VALUE2]'."
                * Extend this pattern as needed for three or more fields, always naming each field and including its specific value.
            * **C. Replace Vague Phrases:** Any generic confirmations like "accept input correctly", "successfully", "correctly", or "as expected" **must be replaced** by explicit statements that include the injected value(s), following the patterns above.
            * **No Data ExpectedResult:** If a step’s `action` was not modified with data, leave its `expectedResult` exactly as is.
            * Only include `expectedResult` keys that were present in the original template. Do not add any new `expectedResult` fields.
4.  **CRITICAL HYGIENE:**
    * Do not add, remove, or reorder any steps.
    * Do not introduce any new keys beyond those in `test_case` (except within existing objects when injecting `value`).
    * Maintain valid JSON throughout.

Now apply exactly that logic to the provided `test_case` and `test_data`. Respond with a single JSON object—no extra chatter.
**CRITICAL OUTPUT REQUIREMENT:**
Respond with ONLY the raw JSON object. No explanations, no markdown blocks, no additional text before or after the JSON. Just the pure JSON object starting with { and ending with }.
"""

    #if USE_ANTHROPIC_CLAUDE:
    #    logger.info("[LLM] Using Anthropic Claude Sonnet 3.7 for test case processing")
    #    response = anthropic_client.messages.create(
    #        model="claude-3-7-sonnet-20250219",
    #        max_tokens=2000,
    #        temperature=0,
    #        system=system_prompt,
    #        messages=[{"role": "user", "content": json.dumps({
    #            "test_case": test_case,
    #            "test_data": test_data_dict
    #        })}]
    #    )
    #    content = response.content[0].text
    #else:
    #    logger.info("[LLM] Using OpenAI GPT-4o-mini for test case processing")
    #    client = OpenAI(api_key=OPENAI_API_KEY)
    #    resp = client.chat.completions.create(
    #        model="gpt-4o-mini",
    #        messages=[
    #            {"role": "system", "content": system_prompt},
    #            {"role": "user", "content": json.dumps({
    #                "test_case": test_case,
    #                "test_data": test_data_dict
    #            })}
    #        ],
    #        temperature=0,
    #        response_format={"type": "json_object"}
    #    )
    #    content = resp.choices[0].message.content
    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": json.dumps({
        "test_case": test_case,
        "test_data": test_data_dict
    })}
]

    content = await call_llm_unified(messages, temperature=0.0)
    print("Response from the LLM: ",content)
    #tc_dict = json.loads(content)
    try:
        tc_dict = json.loads(content)
    except json.JSONDecodeError:
        # Look for JSON in markdown code blocks
        json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
        if json_match:
            json_content = json_match.group(1).strip()
            tc_dict = json.loads(json_content)
        else:
            # Extract JSON object from mixed content
            start = content.find('{')
            if start != -1:
                brace_count = 0
                end = start
                for i, char in enumerate(content[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                
                if end > start:
                    json_content = content[start:end]
                    tc_dict = json.loads(json_content)
                else:
                    logger.error(f"Could not extract valid JSON from LLM response")
                    raise
            else:
                logger.error(f"No JSON found in LLM response")
                raise
    tc = json.dumps(tc_dict, ensure_ascii=False, indent=2)

    print("TC: ",tc)

    test_case_folder = f"{proj_id}/ats/MCP_ATS/Test_Cases/{test_case_number}test_case.json"
    s3.put_object(Body=tc.encode('utf‑8'),
                  Bucket=AWS_BUCKET_NAME,
                  Key=test_case_folder)
    print("Saved TC JSON to: ", test_case_folder)
    # test_case_folder = "Testcases"
    # test_case_path = os.path.join(test_case_folder, f"{test_case_number}test_case.json")

    # with open(test_case_path, "w", encoding="utf-8") as f:
    #     json.dump(tc, f, ensure_ascii=False, indent=2)

    
    steps = [s["action"] for s in tc_dict["steps"]]
    expected_results = [e["expectedResult"] for e in tc_dict["steps"]]
    # print("Expected Results: ",expected_results)
    # print("Stepps: ",steps)

    calls: List[Dict[str,Any]] = []
    req_id = 0
    last_input_values: Dict[str, str] = {}
    async with ws_connect(MCP_URL, subprotocols=["mcp"]) as ws:
        auth_success = await authenticate_with_server(ws)
        if not auth_success:
            logger.error("[AUTH] Authentication failed - cannot proceed")
            return  # Exit if authentication fails
        logger.info("[AUTH] Authentication successful - proceeding with MCP operations")

        # initialize
        req_id += 1
        init_resp = await send_mcp_request(ws, "initialize", {"protocolVersion": "2025-03-26"}, req_id)
        logger.info("MCP initialized.")
        tools_schema = init_resp.result["capabilities"]["tools"] 
        #print("TOOLS USED: ",tools_schema)

        for idx, step in enumerate(steps, start=1):
            logger.info(f"Step {idx}: {step}")
            
            def extract_url(step: str) -> str | None:
                # URL inside single or double quotes
                m = re.search(r"""(['"])(https?://[^'"]+)\1""", step)
                if m:
                    return m.group(2)
                # bare URL fallback
                m = re.search(r"(https?://[^\s\"'<>]+)", step)
                if m:
                    return m.group(1).rstrip(".,);")
                return None
            # ─── Navigation shortcut ──────────────────────────────────────────
            if step.lower().startswith("navigate to the url"):
                url = extract_url(step)
                req_id += 1
                await send_mcp_request(ws, "tools/call",
                                       {"tool_name": "browser_navigate", "tool_args": {"url": url}},
                                       req_id)
                calls.append({"tool": "browser_navigate", "args": {"url": url}})
                req_id += 1
                body_wait_args = {"selectors": ["locator('body')"], "timeout": 30000}
                resp = await send_mcp_request(ws, "tools/call", {
                    "tool_name": "browser_wait_for_selector",
                    "tool_args": body_wait_args
                }, req_id)
                wait_call_entry = {"tool": "browser_wait_for_selector", "args": body_wait_args}
                if resp.result and "actualWaitTime" in resp.result:
                    wait_call_entry["actual_wait_time"] = resp.result["actualWaitTime"]
                    wait_call_entry["succeeded_selector"] = resp.result.get("succeededSelector")
                calls.append(wait_call_entry)
                # --- END: Capture actualWaitTime for browser_wait_for_selector here ---
                await asyncio.sleep(2)
                continue

            # always wait for body
            req_id += 1
            body_wait_args = {"selectors": ["locator('body')"], "timeout": 30000}
            resp = await send_mcp_request(ws, "tools/call", {
                "tool_name": "browser_wait_for_selector",
                "tool_args": body_wait_args
            }, req_id)
            # --- START: Capture actualWaitTime for browser_wait_for_selector here ---
            wait_call_entry = {"tool": "browser_wait_for_selector", "args": body_wait_args}
            if resp.result and "actualWaitTime" in resp.result:
                wait_call_entry["actual_wait_time"] = resp.result["actualWaitTime"]
                wait_call_entry["succeeded_selector"] = resp.result.get("succeededSelector")
            calls.append(wait_call_entry)
            await asyncio.sleep(1)

            # get DOM info
            
            
            async def get_actions_for_step(step,req_id):
                """Fetches DOM info and invokes the LLM. If the LLM returns no actions,
                retries once with a fresh DOM snapshot.
                Returns (actions, elements)."""
                last_elements = []
                for attempt in range(4):
                    # 1) grab fresh DOM
                    req_id += 1
                    resp = await send_mcp_request(
                        ws, "tools/call",
                        {"tool_name": "browser_get_dom_info", "tool_args": {}},
                        req_id
                    )
                    elements = resp.result["elements"]
                    last_elements = elements

                    # 2) ask the LLM
                    actions = await invoke_llm_for_action(step,elements)
                    if actions:
                        return actions, elements, req_id

                    logger.warning(
                        f"Attempt {attempt+1}: no actions for step {step!r}. "
                        f"{'Retrying with fresh DOM...' if attempt == 0 else 'Giving up.'}"
                    )

                # no actions on either try
                return [], last_elements ,req_id


            # — in your main loop —
            actions, elements, req_id = await get_actions_for_step(step,req_id)
            if not actions:
                continue
            # later in your main loop:
            # actions = await get_actions_for_step(step)
            # if not actions:
            #     continue  # skip this step entirely

            # ── Helpers to turn raw selectors into "locator(...)" or "getBy..." ──
            def make_locator(css: str) -> str:
                if "'" in css:
                    return f'locator("{css}")'
                return f"locator('{css}')"

            def build_candidates(raw_list: List[str]) -> List[str]:
                out: List[str] = []
                for expr in raw_list:
                    e = expr.strip()
                    if e.startswith("page."):
                        e = e[len("page."):]
                    if e.startswith("locator(") or e.startswith("getBy"):
                        out.append(e)
                    else:
                        out.append(make_locator(e.strip("'\"")))
                return out
            
            def filter_valid_selectors(selectors: List[str], elements: List[Dict]) -> List[str]:
                """Filter selectors to only include ones that likely exist in the DOM"""
                valid_selectors = []
                actual_ids = set(el.get("id") for el in elements if el.get("id"))

                for sel in selectors:
                    id_match = re.search(r"locator\(['\"]#([^'\"]+)['\"]", sel)
                    if id_match:
                        selector_id = id_match.group(1)
                        if selector_id in actual_ids:
                            valid_selectors.append(sel)
                        else:
                            print(f"Skipping invalid ID selector: {sel} (ID '{selector_id}' not found in DOM)")
                    else:
                        valid_selectors.append(sel)
                return valid_selectors if valid_selectors else selectors
            # execute each action
            for act in actions:
                typ = act["action"]
                tool, tool_args = None, None

                sel_field = act.get("selector", "")
                if isinstance(sel_field, list):
                    raw = sel_field
                else:
                    sel_lit = sel_field.strip()
                    matched = next((el for el in elements
                                    if sel_lit in el.get("selectors", [])
                                    or sel_lit == el.get("selector")), None)
                    raw = matched.get("selectors") if matched else [sel_lit]
                cands = build_candidates(raw)
            
                if typ == "click":
                    tool = "browser_click"
                    filtered_candidates = filter_valid_selectors(cands, elements)
                    tool_args = {"selectors": filtered_candidates}

                elif typ == "assertCellValueInRow":
                    tool = "browser_assert_cell_value_in_row"
                    tool_args = {
                        "table_selector": act.get("table_selector", []), # Expect an array, default to empty list
                        "row_identifier_column": act.get("row_identifier_column"),
                        "row_identifier_value": act.get("row_identifier_value"),
                        "target_column": act.get("target_column"),
                        "expected_value": act.get("expected_value")}
                    

                elif typ == "assertTableColumnValues": # Matches the 'action' key from LLM's JSON
                    tool = "browser_assert_table_column_values"
                    raw_table_selectors = act.get("table_selector", [])
                    processed_table_selectors = build_candidates(raw_table_selectors) # Process table selectors
                    assertion_type_from_llm = act.get("assertion_type")
                    if assertion_type_from_llm:
                        final_assertion_type = assertion_type_from_llm
                    elif act.get("negative_assertion") is True:
                        final_assertion_type = "none"
                    else:
                        final_assertion_type = "all" # Default if nothing specified (for backward compatibility)

                    tool_args = {
                        "table_selector": processed_table_selectors,
                        "column_header": act.get("column_header"),
                        "expected_value": act.get("expected_value"),
                        "match_type": act.get("match_type", "includes"),
                        "assertion_type": final_assertion_type  # Default to 'includes' if not specified by LLM
                    }


                elif typ == "browser_assert_filtered_table_rows": # The action name from LLM
                    tool = "browser_assert_filtered_table_rows"
                    raw_tbl = act.get("table_selector", [])
                    processed_table_selectors = build_candidates(raw_tbl) # Normalize selectors

                    # Ensure filter_conditions is a list of dicts
                    filter_conditions = act.get("filter_conditions", [])
                    if not isinstance(filter_conditions, list):
                        logger.warning(f"LLM returned unexpected type for 'filter_conditions': {type(filter_conditions)}. Expected list. Skipping this verification.")
                        continue
                    
                    tool_args = {
                        "table_selector": processed_table_selectors,
                        "filter_conditions": filter_conditions,
                        "assert_column": act.get("assert_column"),
                        "assert_value": act.get("assert_value"),
                        "assert_match_type": act.get("assert_match_type", "includes"),
                        "assert_negative_assertion": act.get("assert_negative_assertion", False)
                    }
                elif typ in ("fill", "type"):
                    tool = "browser_type"
                    text_to_fill = act.get("text", "")
                    tool_args = {"selectors": cands, "text": text_to_fill}
                    if cands:
                        normalized_key = normalize_selector(cands[0]) 
                        last_input_values[normalized_key] = text_to_fill
                        logger.debug(f"Stored input for '{normalized_key}' (original: '{cands[0]}'): '{text_to_fill}'")

                elif typ == "assertUrlContains":
                    tool = "browser_assert_url_contains"
                    tool_args = {"fragment": act.get("fragment", "")}

                elif typ in ("assertValue", "assert_value"):
                    tool = "browser_assert_value"
                # reuse the exact same candidate list
                    tool_args = {
                        "selector": cands,
                        "expected": act.get("expected", "")
                    }
                elif typ in ("assertSelectedOption","assert_selected_option"):
                    tool = "browser_assert_selected_option"
                # again reuse the same normalized selector
                    tool_args = {
                        "selector": cands,
                        "expected": act.get("expected", "")
                    }

                elif typ in ("assertDisplayedOptionText", "assert_displayed_option_text"):
                    tool = "browser_assert_displayed_option_text"
                    tool_args = {
                        "selectors": cands,
                        "expected": act.get("expected", ""),
                    }
                elif typ in ("assertElementVisible", "assert_element_visible"):
                    tool = "browser_assert_element_visible"
                    tool_args = {"selectors": cands}
                    


                elif typ in ("select", "selectOption"):
                    # Use the first candidate selector (or directly the sel_field if it's already a good Playwright selector string)
                    # The tool will handle eval'ing this selector.
                    target_selector = cands[0] if cands else sel_field 
                    option_value = act.get("value", "") # This is the option text for the tool
                    # and option_value:
                    if target_selector: 
                        tool = "browser_select_option" # Use the name of your new server-side tool
                        tool_args = {
                            "selector": target_selector, # Pass the Playwright selector string
                            "value": option_value        # Pass the option text
                        }
                        normalized_key = normalize_selector(target_selector)
                    
                        last_input_values[normalized_key] = option_value
                        # --- END: STORE SELECTED VALUE ---`
                    else:
                        logger.warning(f"Skipping select action due to missing selector or value: {act}")
                        continue


                elif typ == "check":
                    tool = "browser_check"
                    tool_args = {"selector": cands}
                elif typ == "uncheck":
                    tool = "browser_uncheck"
                    tool_args = {"selector": cands}
                elif typ == "assertText":
                    tool = "browser_assert_text_visible"
                    tool_args = {"selectors": cands, "text": act.get("text","")}
                elif typ == "waitForSelector":
                    tool = "browser_wait_for_selector"
                    tool_args = {"selectors": cands, "timeout": 30000}
                elif typ == "navigate":
                    url = act.get("url","")
                    if url.lower().startswith("http"):
                        tool = "browser_navigate"
                        tool_args = {"url": url}
                    else:
                        sel = make_locator(f'a:has-text("{url}")')
                        tool = "browser_click"
                        tool_args = {"selectors":[sel]}
                else:
                    logger.warning(f"Unsupported action {typ!r}. Skipping.")
                    continue

                #if tool == "browser_click":
                #    sels = tool_args["selectors"]
                #    css = sels[0]
                #    # look for a second selector that carries text
                #    text_sel = next((s for s in sels[1:]
                #                    if s.startswith("getByText(") or s.startswith("getByRole(")), None)
                #    if css.startswith("locator('button") and text_sel:
                #        m2 = re.match(r".*?['\"](.+?)['\"]\)", text_sel)
                #        label = m2.group(1) if m2 else None
                #        if label:
                #            composite = f"{css}.filter({{ hasText: '{label}' }})"
                #            tool_args["selectors"] = [composite]
#
                #    # now pre-wait on *all* candidate selectors, in priority order
                #    req_id += 1
                #
#
#
                #req_id += 1
                #resp = await send_mcp_request(ws, "tools/call",
                #                                {"tool_name": tool, "tool_args": tool_args},
                #                                req_id)
#
                #call_entry = {"tool":tool,"args":tool_args}
                
                #try:
                #    if tool == "browser_click":
                #        sels = tool_args["selectors"]
                #        click_success = False
        #
                #        # Try each selector until one works
                #        for css in sels:
                #            try:
                #                # Create individual tool_args for this selector
                #                single_selector_args = {"selectors": [css]}
                #                
                #                # Look for a second selector that carries text (keep existing logic)
                #                text_sel = next((s for s in sels[1:] 
                #                                 if s.startswith("getByText(") or s.startswith("getByRole(")), None)
                #                if css.startswith("locator('button") and text_sel:
                #                    m2 = re.match(r".*?['\"](.+?)['\"]\)", text_sel)
                #                    label = m2.group(1) if m2 else None
                #                    if label:
                #                        composite = f"{css}.filter({{ hasText: '{label}' }})"
                #                        single_selector_args["selectors"] = [composite]
                #    
                #                req_id += 1
                #                resp = await send_mcp_request(ws, "tools/call",
                #                                                {"tool_name": tool, "tool_args": single_selector_args},
                #                                                req_id)
                #
                #                # Check if this selector worked
                #                if resp.result and resp.result.get("success"):
                #                    click_success = True
                #                    tool_args = single_selector_args  # Update tool_args to the successful one
                #                    break
                #    
                #            except Exception as e:
                #                logger.warning(f"Click selector {css} failed: {e}")
                #                continue
        #
                #        if not click_success:
                #            logger.warning(f"All click selectors failed for step")
                #            # Use original tool_args as fallback
                #    else:
                #        req_id += 1
                #        resp = await send_mcp_request(ws, "tools/call",
                #                                        {"tool_name": tool, "tool_args": tool_args},
                #                                        req_id)
                #    
                #    call_entry = {"tool": tool, "args": tool_args}
                #    calls.append(call_entry)
#
                #except RuntimeError as e:
                #    logger.error(f"Action failed for step {idx}: {tool} - {e}")
                #    print(f"ACTION FAILED: Step {idx}, Tool '{tool}' - {e}")
                #    failed_call_entry = {"tool": tool, "args": tool_args, "status": "failed", "error": str(e)}
                #    calls.append(failed_call_entry)
                #    continue  # Skip to next action
                #except Exception as e:
                #    logger.error(f"Unexpected action error for step {idx}: {tool} - {e}", exc_info=True)
                #    print(f"UNEXPECTED ACTION ERROR: Step {idx}, Tool '{tool}' - {e}")
                #    failed_call_entry = {"tool": tool, "args": tool_args, "status": "error", "error": str(e)}
                #    calls.append(failed_call_entry)
                #    continue
#
                ## --- START: Capture actualWaitTime for browser_wait_for_selector when called as a general action ---
                #if tool == "browser_wait_for_selector" and resp.result and "actualWaitTime" in resp.result:
                #    call_entry["actual_wait_time"] = resp.result["actualWaitTime"]
                #    call_entry["succeeded_selector"] = resp.result.get("succeededSelector")
                ## --- END: Capture actualWaitTime for browser_wait_for_selector when called as a general action ---
                #if isinstance(resp.result, dict) and "succeededSelector" in resp.result:
                #    call_entry["succeeded_selector"] = resp.result["succeededSelector"]
                #calls.append(call_entry)
                #req_id += 1
                #body_wait_args = {"selectors": ["locator('body')"], "timeout": 30000}
                #wait_resp = await send_mcp_request(ws, "tools/call", {
                #    "tool_name": "browser_wait_for_selector",
                #    "tool_args": body_wait_args
                #}, req_id)
                ## Capture wait_resp details as well for complete logging
                #wait_call_entry = {"tool": "browser_wait_for_selector", "args": body_wait_args}
                #if wait_resp.result and isinstance(wait_resp.result, dict) and "actualWaitTime" in wait_resp.result:
                #    wait_call_entry["actual_wait_time"] = wait_resp.result["actualWaitTime"]
                #    wait_call_entry["succeeded_selector"] = wait_resp.result.get("succeededSelector")
                #calls.append(wait_call_entry)
                #await asyncio.sleep(5)
            try:
                if tool == "browser_click":
                    sels = tool_args["selectors"]
                    click_success = False
   
        # Try each selector until one works
                    for css in sels:
                        try:
                            # Create individual tool_args for this selector
                            single_selector_args = {"selectors": [css]}
                            
                            # Look for a second selector that carries text (keep existing logic)
                            text_sel = next((s for s in sels[1:] 
                                             if s.startswith("getByText(") or s.startswith("getByRole(")), None)
                            if css.startswith("locator('button") and text_sel:
                                m2 = re.match(r".*?['\"](.+?)['\"]\)", text_sel)
                                label = m2.group(1) if m2 else None
                                if label:
                                    composite = f"{css}.filter({{ hasText: '{label}' }})"
                                    single_selector_args["selectors"] = [composite]
                    
                            req_id += 1
                            resp = await send_mcp_request(ws, "tools/call",
                                                            {"tool_name": tool, "tool_args": single_selector_args},
                                                            req_id)
                    
                            # Check if this selector worked
                            if resp.result and resp.result.get("success"):
                                click_success = True
                                tool_args = single_selector_args  # Update tool_args to the successful one
                                break
                    
                        except Exception as e:
                            logger.warning(f"Click selector {css} failed: {e}")
                            continue
            
                    if not click_success:
                        logger.warning(f"All click selectors failed for step")
            # Use original tool_args as fallback
                else:
                    req_id += 1
                    resp = await send_mcp_request(ws, "tools/call",
                                    {"tool_name": tool, "tool_args": tool_args},
                                    req_id)
    
    # Create call entry and handle success
                call_entry = {"tool": tool, "args": tool_args}
    
    # Capture additional response data if available
                if 'resp' in locals() and isinstance(resp.result, dict):
                    if tool == "browser_wait_for_selector" and "actualWaitTime" in resp.result:
                        call_entry["actual_wait_time"] = resp.result["actualWaitTime"]
                        call_entry["succeeded_selector"] = resp.result.get("succeededSelector")
                    if "succeededSelector" in resp.result:
                        call_entry["succeeded_selector"] = resp.result["succeededSelector"]
                
                calls.append(call_entry)
    
    # Post-action wait
                req_id += 1
                body_wait_args = {"selectors": ["locator('body')"], "timeout": 30000}
                wait_resp = await send_mcp_request(ws, "tools/call", {
                    "tool_name": "browser_wait_for_selector",
                    "tool_args": body_wait_args
                }, req_id)
                
                # Capture wait_resp details as well for complete logging
                wait_call_entry = {"tool": "browser_wait_for_selector", "args": body_wait_args}
                if wait_resp.result and isinstance(wait_resp.result, dict) and "actualWaitTime" in wait_resp.result:
                    wait_call_entry["actual_wait_time"] = wait_resp.result["actualWaitTime"]
                    wait_call_entry["succeeded_selector"] = wait_resp.result.get("succeededSelector")
                calls.append(wait_call_entry)
                
                await asyncio.sleep(5)

            except RuntimeError as e:
                logger.error(f"Action failed for step {idx}: {tool} - {e}")
                print(f"ACTION FAILED: Step {idx}, Tool '{tool}' - {e}")
                failed_call_entry = {"tool": tool, "args": tool_args, "status": "failed", "error": str(e)}
                calls.append(failed_call_entry)
                continue  # Skip to next action
            except Exception as e:
                logger.error(f"Unexpected action error for step {idx}: {tool} - {e}", exc_info=True)
                print(f"UNEXPECTED ACTION ERROR: Step {idx}, Tool '{tool}' - {e}")
                failed_call_entry = {"tool": tool, "args": tool_args, "status": "error", "error": str(e)}
                calls.append(failed_call_entry)
                continue  # Skip to next action

#-------------------Verification-----------------------
            print("--------Calling DOM for verification--------")
            req_id += 1
            resp = await send_mcp_request(ws, "tools/call",
                {"tool_name":"browser_get_dom_info","tool_args":{}}, req_id)
            post_elements = resp.result["elements"]
            used_locators = cands.copy()
            print("------------------USED Selectors---------------:")
            print("Cands: ",used_locators)        # 5) ask LLM for the single verification object
            expected = tc_dict["steps"][idx-1]["expectedResult"]
            logger.info(f"Expected Result {idx}: {expected}")
            llm_verification_response  = await invoke_llm_for_verification(
                    expectedResult=expected,
                    used_locators=used_locators,   # from step 3
                    elements=post_elements
                )
            if isinstance(llm_verification_response, dict):
    # LLM returned a single dictionary, wrap it in a list
                verify_objects = [llm_verification_response]
            elif isinstance(llm_verification_response, list):
                # LLM returned a list, as expected by the prompt's main instruction
                verify_objects = llm_verification_response
            else:
                
                logger.error(f"Unexpected type returned from invoke_llm_for_verification: {type(llm_verification_response)}. Expected dict or list.")
                raise TypeError("LLM verification response was not a dictionary or a list.")

                

            for verify_obj in verify_objects:
                typ = verify_obj["action"]
                tool_args = {} # Initialize tool_args here to avoid potential issues

                if typ == "assertUrlContains":
                    tool = "browser_assert_url_contains"
                    tool_args = {"fragment": verify_obj["fragment"]}
                
                elif typ in ("assertValue", "assert_value"):
                    tool = "browser_assert_value"
                    # The LLM will now provide 'selector' as a list of strings
                    selectors_list = verify_obj["selector"] 
                    
                    # Ensure it's always a list, even if LLM somehow sends a string (though prompt should prevent this)
                    if isinstance(selectors_list, str):
                        selectors_list = [selectors_list]
                    elif not isinstance(selectors_list, list):
                        logger.warning(f"LLM returned unexpected type for 'selector' in assertValue: {type(selectors_list)}. Expected list. Skipping this verification.")
                        continue # Skip if not a list or string

                    if not selectors_list: # Handle empty list
                        logger.warning("LLM returned an empty list for selectors in assertValue. Skipping this verification.")
                        continue

                    # --- START: OVERRIDE EXPECTED VALUE FROM STORED INPUTS ---
                    expected_value_for_assertion = verify_obj.get("expected", "") # Default to LLM's guess or what it derived
                    
                    # Iterate through the selectors the LLM provided for this assertion
                    # and check if any were recently input targets.
                    # for sel in selectors_list:
                    #     if sel in last_input_values:
                    #         expected_value_for_assertion = last_input_values[sel]
                    #         logger.info(f"Overriding expected value for '{sel}' to '{expected_value_for_assertion}' (from last_input_values).")
                    #         break # Found a match, use this and stop looking
                    # --- END: OVERRIDE EXPECTED VALUE ---

                    # print("Selector List: ",selectors_list)
                    # print("Expected Value Befor: ",expected_value_for_assertion)
                    # print("Last input: ",last_input_values)
                    # found_override = False
                    # for sel_from_llm in selectors_list: # Iterate through all selectors LLM provided for verification
                    #     normalized_sel_from_llm = normalize_selector(sel_from_llm) # Check for exact match
                    #     if normalized_sel_from_llm in last_input_values: # Check against our stored, normalized keys
                    #         expected_value_for_assertion = last_input_values[normalized_sel_from_llm]
                    #         logger.info(f"Overriding expected value for '{sel_from_llm}' (normalized to '{normalized_sel_from_llm}') to '{expected_value_for_assertion}' (from last_input_values).")
                    #         found_override = True
                    #         break

                    tool_args = {
                        "selectors": selectors_list, # Pass the array of selectors
                        "expected": expected_value_for_assertion 
                    }
                
                # ... (assertText remains the same as it already expects a list for 'selectors')
                elif typ in ("assertSelectedOption","assert_selected_option"):
                    tool = "browser_assert_selected_option"
                    selector_value = verify_obj["selector"]
                    # Ensure selector is a string for assertSelectedOption
                    if isinstance(selector_value, list):
                        # If LLM returns a list for assertSelectedOption, take the first element
                        selector_value = selector_value[0] if selector_value else None
                        if selector_value is None:
                            logger.warning("LLM returned empty list for selector in assertSelectedOption. Skipping this verification.")
                            continue
                    # --- START: OVERRIDE EXPECTED VALUE FROM STORED INPUTS ---
                    expected_value_for_assertion = verify_obj.get("expected", "") # Default to LLM's guess
                    # if selector_value:
                    #     normalized_sel_from_llm = normalize_selector(selector_value)
                    #     if normalized_sel_from_llm in last_input_values:
                    #         expected_value_for_assertion = last_input_values[normalized_sel_from_llm]
                    #         logger.info(f"Overriding expected selected option for '{selector_value}' (normalized to '{normalized_sel_from_llm}') to '{expected_value_for_assertion}' (from last_input_values).")

                    tool_args = {
                        "selector": selector_value,
                        "expected": expected_value_for_assertion 
                    }
                elif typ in ("assertCellValueInRow", "assert_cell_value_in_row"):
                    tool = "browser_assert_cell_value_in_row"
                    raw_tbl = verify_obj.get("table_selector", [])
                    if isinstance(raw_tbl, str):
                        raw_tbl = [raw_tbl]
                    processed_tbl = build_candidates(raw_tbl)
                    tool_args = {
                        "table_selector": processed_tbl,
                        "row_identifier_column": verify_obj.get("row_identifier_column"),
                        "row_identifier_value": verify_obj.get("row_identifier_value"),
                        "target_column": verify_obj.get("target_column"),
                        "expected_value": verify_obj.get("expected_value")
                    }

                elif typ == "browser_assert_filtered_table_rows": # The action name from LLM
                    tool = "browser_assert_filtered_table_rows"
                    raw_tbl = verify_obj.get("table_selector", [])
                    if isinstance(raw_tbl, str):
                        raw_tbl = [raw_tbl]
                    processed_tbl = build_candidates(raw_tbl) # Normalize selectors

                    # Ensure filter_conditions is a list of dicts
                    filter_conditions = verify_obj.get("filter_conditions", [])
                    if not isinstance(filter_conditions, list):
                        logger.warning(f"LLM returned unexpected type for 'filter_conditions': {type(filter_conditions)}. Expected list. Skipping this verification.")
                        continue
                    
                    tool_args = {
                        "table_selector": processed_tbl,
                        "filter_conditions": filter_conditions,
                        "assert_column": verify_obj.get("assert_column"),
                        "assert_value": verify_obj.get("assert_value"),
                        "assert_match_type": verify_obj.get("assert_match_type", "includes"),
                        "assert_negative_assertion": verify_obj.get("assert_negative_assertion", False)
                    }

                elif typ in ("assertTableColumnValues", "assert_table_column_values"):
                    tool = "browser_assert_table_column_values"
                    raw_tbl = verify_obj.get("table_selector", [])
                    if isinstance(raw_tbl, str):
                        raw_tbl = [raw_tbl]
                    processed_tbl = build_candidates(raw_tbl)
                    assertion_type_from_llm = verify_obj.get("assertion_type")
                    if assertion_type_from_llm:
                        final_assertion_type = assertion_type_from_llm
                    elif act.get("negative_assertion") is True:
                        final_assertion_type = "none"
                    else:
                        final_assertion_type = "all"
                    tool_args = {
                        "table_selector": processed_tbl,
                        "column_header": verify_obj.get("column_header"),
                        "expected_value": verify_obj.get("expected_value"),
                        "match_type": verify_obj.get("match_type", "includes"),
                        "assertion_type": final_assertion_type
                    }
                
                elif typ == "assertDisplayedOptionText": 
                    tool = "browser_assert_displayed_option_text" # New tool name
                    selectors_list = verify_obj["selector"] 
                    
                    # Ensure 'selectors' for browser_assert_displayed_option_text is ALWAYS a list
                    if isinstance(selectors_list, str):
                        selectors_list = [selectors_list]
                    elif not isinstance(selectors_list, list):
                        logger.warning(f"LLM returned unexpected type for 'selector' in assertDisplayedOptionText: {type(selectors_list)}. Expected list. Skipping this verification.")
                        continue # Skip if not a list or string

                    if not selectors_list: # Handle empty list
                        logger.warning("LLM returned an empty list for selectors in assertDisplayedOptionText. Skipping this verification.")
                        continue

                    # Override expected value from last_input_values, similar to assertValue
                    expected_value_for_assertion = verify_obj.get("expected", "") 
                    # found_override = False # This variable isn't strictly necessary if you just break, but useful for debugging/logging
                    # for sel_from_llm in selectors_list:
                    #     normalized_sel_from_llm = normalize_selector(target_selector)
                    #     expected_value_for_assertion = last_input_values[normalized_sel_from_llm]
                    #     # if normalized_sel_from_llm in last_input_values:
                            
                    #     #     logger.info(f"Overriding expected displayed option text for '{sel_from_llm}' (normalized to '{normalized_sel_from_llm}') to '{expected_value_for_assertion}' (from last_input_values).")
                    #     #     found_override = True
                    #     #     break

                    tool_args = {
                        "selectors": selectors_list, # Pass the array of selectors
                        "expected": expected_value_for_assertion 
                    }

                elif typ == "assertElementVisible":
                    tool = "browser_assert_element_visible"
                    selector_value = verify_obj["selector"]
                    # Ensure selector is a single string for assertElementVisible
                    if isinstance(selector_value, list):
                        selector_value = selector_value[0] if selector_value else None
                        if selector_value is None:
                            logger.warning("LLM returned empty list for selector in assertElementVisible. Skipping this verification.")
                            continue
                    
                    tool_args = {
                        "selector": selector_value
                        # 'timeout' is optional, can add if needed: "timeout": 5000
                    }
                else:  # assertText
                    tool = "browser_assert_text_visible"
                    selector_value = verify_obj["selector"]
                    # Ensure 'selectors' for browser_assert_text_visible is ALWAYS a list
                    if isinstance(selector_value, str):
                        corrected_selectors = [selector_value]
                    elif isinstance(selector_value, list):
                        corrected_selectors = selector_value
                    else:
                        logger.warning(f"LLM returned unexpected type for 'selector' in assertText: {type(selector_value)}. Expected string or list. Skipping this verification.")
                        continue

                    tool_args = {
                        "selectors": corrected_selectors, # This is now guaranteed to be a list
                        "text": verify_obj["text"]
                    }
                
                # req_id += 1
                # await send_mcp_request(ws, "tools/call",
                #     {"tool_name":tool, "tool_args":tool_args}, req_id)
                # calls.append({"tool": tool, "args": tool_args})
                try:
                    req_id += 1
                    resp = await send_mcp_request(ws, "tools/call",
                                                  {"tool_name":tool, "tool_args":tool_args}, req_id)
                    call_entry = {"tool":tool,"args":tool_args}
                    # Check if the tool execution itself reported an error from MCP
                    if resp.error:
                        error_message = resp.error.get('message', 'Unknown tool execution error.')
                        logger.error(f"MCP tool '{tool}' failed for verification: {error_message}")
                        print(f"VERIFICATION FAILED: Tool '{tool}' - {error_message}")
                    else:
                        # Log success message if no error
                        logger.info(f"MCP tool '{tool}' succeeded for verification.")
                        print(f"VERIFICATION PASSED: Tool '{tool}'")

                    if isinstance(resp.result, dict) and "succeededSelector" in resp.result:
                        call_entry["succeeded_selector"] = resp.result["succeededSelector"]
                    calls.append(call_entry) # Still log the call

                #except RuntimeError as e:
                #    # This catches errors where send_mcp_request raises an exception
                #    # (e.g., if the MCP itself is unreachable or has a fundamental issue with the request)
                #    logger.error(f"Failed to send/receive MCP request for tool '{tool}' during verification: {e}")
                #    print(f"CRITICAL VERIFICATION ERROR: Could not execute tool '{tool}' - {e}")
                #    # The program will continue to the next iteration of the loop
                #    # or the next test step after this block.
                #except Exception as e:
                #    # Catch any other unexpected errors during the process within this try block
                #    logger.error(f"An unexpected error occurred during verification for tool '{tool}': {e}", exc_info=True)
                #    print(f"UNEXPECTED VERIFICATION ERROR: Tool '{tool}' - {e}")
                except RuntimeError as e:
                    logger.error(f"Verification failed for step {idx}: {tool} - {e}")
                    print(f"VERIFICATION FAILED: Step {idx}, Tool '{tool}' - {e}")
                    failed_call_entry = {"tool": tool, "args": tool_args, "status": "verification_failed", "error": str(e)}
                    calls.append(failed_call_entry)
                    continue  # Continue to next verification instead of crashing
                except Exception as e:
                    logger.error(f"Unexpected verification error for step {idx}: {tool} - {e}", exc_info=True)
                    print(f"UNEXPECTED VERIFICATION ERROR: Step {idx}, Tool '{tool}' - {e}")
                    failed_call_entry = {"tool": tool, "args": tool_args, "status": "verification_error", "error": str(e)}
                    calls.append(failed_call_entry)
                    continue
                
    
    print("Calls: ",calls)
    

   

  
    def generate_java_playwright(calls, test_name="AutoGeneratedTest"):
        def _parse_selector(sel: str):
            # --- Accept lists/tuples/None safely ---
            if isinstance(sel, (list, tuple)):
                sel = sel[0] if sel else ""
            elif sel is None:
                sel = ""
            else:
                sel = str(sel)

            m_loc  = re.match(r"""locator\(['"](.+?)['"]\)""", sel)
            m_text = re.match(r"""page\.getByText\(['"](.+?)['"]\)""", sel)
            m_role = re.match(r"""page\.getByRole\((.+)\)""", sel)
            if m_text:
                txt = m_text.group(1).replace('"', '\\"')
                return f'page.getByText("{txt}")', txt
            if m_loc:
                loc = m_loc.group(1).replace('"', '\\"')
                return f'page.locator("{loc}")', loc
            if m_role:
                return f'page.getByRole({m_role.group(1)})', None
            if sel.startswith("page."):
                return sel, None
            escaped = sel.replace('"', '\\"')
            return f'page.locator("{escaped}")', escaped

        lines = [
            "import com.microsoft.playwright.*;",
            "import org.junit.jupiter.api.*;",
            "",
            "import static org.junit.jupiter.api.Assertions.*;",
            "",
            f"public class {test_name} {{",
            "    private static Playwright playwright;",
            "    private static Browser browser;",
            "    private static Page page;",
            "",
            "    @BeforeAll",
            "    public static void setUp() {",
            "        playwright = Playwright.create();",
            "        browser = playwright.chromium().launch(",
            "            new BrowserType.LaunchOptions().setHeadless(false));",
            "        page = browser.newPage();",
            "    }",
            "",
            "    @Test",
            f"    public void {test_name.lower()}() {{"
        ]

        for c in calls:
            t = c['tool']
            a = c['args']

            if not t or not a:
                continue

            if not isinstance(t, str):
                continue
            
            sel = None
            if 'succeeded_selector' in c:
                sel = c['succeeded_selector']
            elif 'selectors' in a and a['selectors']:
                sel = a['selectors'][0]
            elif 'selector' in a:
                sel = a['selector']

            pl_expr, raw_sel = (None, None)
            if sel:
                pl_expr, raw_sel = _parse_selector(sel)

            if t == "browser_navigate":
                lines.append(f'        page.navigate("{a.get("url","")}");')

            elif t == "browser_click" and pl_expr:
                lines.append(f"        {pl_expr}.click();")

            elif t == "browser_type" and pl_expr:
                txt = a.get("text","").replace('"','\\"')
                lines.append(f'        {pl_expr}.fill("{txt}");')

            elif t == "browser_select_option" and pl_expr:
                val = a.get("value","").replace('"','\\"')
                lines.append(f'        {pl_expr}.selectOption("{val}");')

            elif t == "browser_check" and pl_expr:
                lines.append(f"        {pl_expr}.check();")

            elif t == "browser_uncheck" and pl_expr:
                lines.append(f"        {pl_expr}.uncheck();")

            elif t == "browser_wait_for_selector" and raw_sel:
                timeout = a.get("timeout",30000)
                lines.extend([
                    f'        page.waitForSelector("{raw_sel}",',
                    f"            new Page.WaitForSelectorOptions().setTimeout({timeout}));"
                ])

            elif t == "browser_assert_value" and pl_expr:
                exp = a.get("expected","").replace('"','\\"')
                lines.extend([
                    f'        assertEquals("{exp}",',
                    f"            {pl_expr}.inputValue(),",
                    f'            "Expected value in {pl_expr} to be \'{exp}\'");'
                ])

            elif t == "browser_assert_text_visible" and pl_expr:
                txt = a.get("text","").replace('"','\\"')
                lines.extend([
                    f'        assertEquals("{txt}",',
                    f"            {pl_expr}.innerText(),",
                    f'            "Expected text at {pl_expr} to be \'{txt}\'");'
                ])

            elif t == "browser_assert_element_visible" and pl_expr:
                lines.extend([
                    "        assertTrue(",
                    f"            {pl_expr}.isVisible(),",
                    f'            "Expected element to be visible: {pl_expr}");'
                ])

            elif t == "browser_assert_url_contains":
                frag = a.get("fragment","").replace('"','\\"')
                lines.extend([
                    "        assertTrue(",
                    f'            page.url().contains("{frag}"),',
                    f'            "Expected URL to contain \'{frag}\', but was " + page.url());'
                ])

            elif t == "browser_assert_selected_option" and pl_expr:
                exp = a.get("expected","").replace('"','\\"')
                lines.extend([
                    f'        assertEquals("{exp}",',
                    f"            {pl_expr}.inputValue(),",
                    f'            "Expected selected option value in {pl_expr} to be \'{exp}\'");'
                ])

            elif t == "browser_assert_displayed_option_text" and pl_expr:
                exp = a.get("expected","").replace('"','\\"')
                lines.extend([
                    f'        assertEquals("{exp}",',
                    f'            {pl_expr}.locator("option:checked").innerText(),',
                    f'            "Expected displayed option text in {pl_expr} to be \'{exp}\'");'
                ])

            # ----- Table / grid assertions -----
            elif t == "browser_assert_cell_value_in_row":
                # Always take the first item from table_selector
                tbl_sel = a["table_selector"][0] if a.get("table_selector") else ""
                row_col = a.get("row_identifier_column")
                row_val = a.get("row_identifier_value","").replace('"','\\"')
                tgt_col = a.get("target_column")
                exp_val = a.get("expected_value","").replace('"','\\"')
                tbl_loc,_ = _parse_selector(tbl_sel) if tbl_sel else ("page.locator(\"table\")",None)
                lines.extend([
                    f'        Locator row = {tbl_loc}.locator("tbody tr").filter(new Locator.FilterOptions()',
                    f'            .setHasText("{row_val}"));',
                    f'        String cellText = row.locator("td").nth({tgt_col}).innerText();',
                    f'        assertEquals("{exp_val}", cellText,',
                    f'            "Expected cell value in column {tgt_col} for row with {row_col}=\'{row_val}\'");'
                ])

            elif t == "browser_assert_table_column_values":
                # Always take the first item from table_selector
                tbl_sel = a["table_selector"][0] if a.get("table_selector") else ""
                col_hdr = a.get("column_header","").replace('"','\\"')
                exp_val = a.get("expected_value","").replace('"','\\"')
                match_type = a.get("match_type","includes")
                assertion_type = a.get("assertion_type","all")
                tbl_loc,_ = _parse_selector(tbl_sel) if tbl_sel else ("page.locator(\"table\")",None)
                lines.extend([
                    f'        Locator cells = {tbl_loc}.locator("tbody td:nth-child(" +',
                    f'            ({tbl_loc}.locator("thead th").allInnerTexts().indexOf("{col_hdr}") + 1) + ")");',
                    "        for (String v : cells.allInnerTexts()) {",
                    f'            boolean ok = "{match_type}".equals("includes") ?',
                    f'                v.contains("{exp_val}") : v.equals("{exp_val}");',
                    f'            if ("{assertion_type}".equals("none")) assertFalse(ok);',
                    f'            else if ("{assertion_type}".equals("any")) {{ if (ok) break; }}',
                    f'            else assertTrue(ok);',
                    "        }"
                ])

            elif t == "browser_assert_filtered_table_rows":
                # Always take the first item from table_selector  (FIXED)
                tbl_sel = a["table_selector"][0] if a.get("table_selector") else ""
                assert_col = a.get("assert_column")
                assert_val = a.get("assert_value","").replace('"','\\"')
                match_type = a.get("assert_match_type","includes")
                neg = a.get("assert_negative_assertion",False)
                tbl_loc,_ = _parse_selector(tbl_sel) if tbl_sel else ("page.locator(\"table\")",None)
                lines.extend([
                    f'        int colIdx = {tbl_loc}.locator("thead th").allInnerTexts().indexOf("{assert_col}");',
                    "        for (Locator row : " + tbl_loc + ".locator('tbody tr').all()) {",
                    "            String cell = row.locator('td').nth(colIdx).innerText();",
                    f'            boolean ok = "{match_type}".equals("includes") ?',
                    f'                cell.contains("{assert_val}") : cell.equals("{assert_val}");',
                    f'            if ({str(neg).lower()}) assertFalse(ok); else assertTrue(ok);',
                    "        }"
                ])

        lines.extend([
            "    }",
            "",
            "    @AfterAll",
            "    public static void tearDown() {",
            "        if (browser != null) browser.close();",
            "        if (playwright != null) playwright.close();",
            "    }",
            "}"
        ])

        return "\n".join(lines)


        
    java_code = generate_java_playwright(calls)
    target_folder = f"{proj_id}/ats/MCP_ATS/Playwright_Java/src/test/java/com/cognitest/testcases{test_case_number}MCP_test_script.java"
    #target_folder = f"{proj_id}/ats/MCP_ATS/MCP_test_script/{test_case_number}MCP_test_script.java"
    # os.makedirs(target_folder, exist_ok=True)
    # mcp_test_script = os.path.join(target_folder, f"{test_case_id}MCP_test_script.java")
    # with open(mcp_test_script, "w", encoding="utf-8") as f:
    #     f.write(java_code)
    s3.put_object(Body=java_code.encode('utf‑8'),
                  Bucket=AWS_BUCKET_NAME,
                  Key=target_folder)

    print(f"Playwright_Java test saved to {target_folder}")
    # print(f" Java+Playwright test saved to ")


    print("----------Updating Menupage.java-------------")
    # INPUT_POM_PATH = "NewProject/NewProject/src/test/java/com/page/base/MenuPage.java"
    INPUT_TEST_PATH = target_folder
    WRAPPER_JSON_PATH = f"{proj_id}/ats/MCP_ATS/playwright_wrapper_methods_clean.json"

    print("Input POM path: ",input_pom_path)
    print("Project Wrapper JSON: ",WRAPPER_JSON_PATH)
    generate_or_update_pom(
        pom_path=input_pom_path,
        test_path=INPUT_TEST_PATH,
        wrapper_json_path=WRAPPER_JSON_PATH
    )

    print("---------------------MenuPage.java Updated--------------------")
    print("------------------------Generating ATS for test-Case---------------")
    
    refresh_test_data = f"{proj_id}/ats/MCP_ATS/Playwright_Java/src/test/java/com/cognitest/testcases/TestData/testData_{test_case_number}.json"
    s3.put_object(Body=test_data,
                  Bucket=AWS_BUCKET_NAME,
                  Key=refresh_test_data)
    
    process_ats(
    input_test_script_path=target_folder,
    input_test_case_path=test_case_folder,
    input_pom_path=input_pom_path,
    test_data=test_data_dict,
    refresh_test_data = refresh_test_data, # This is the input to the generator
    output_ats_path=f"{proj_id}/ats/MCP_ATS/Playwright_Java/src/test/java/com/cognitest/testcases/TC_{test_case_number}.java"
)

#     print("------------------------ATS Generated-------------------")

    
    
if __name__ == "__main__":
    asyncio.run(main())

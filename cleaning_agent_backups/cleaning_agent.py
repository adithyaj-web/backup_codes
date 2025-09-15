import os, json, json5, base64, mimetypes
import json
import re
from typing import Dict, Any, List, Optional
from openai import OpenAI
from dotenv import load_dotenv
import requests
import logging
import anthropic
import boto3

load_dotenv()
#OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Global variables for LLM providers
llm_provider = ""
client = None
anthropic_client = None

USE_LOCAL_IMAGES = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#--------------LLM PROVIDER AND API SETUP INTEGRATIONS--------------------
def get_llm_provider_for_project(project_id: int) -> str:
    url = "http://localhost:3000/api/getLlmProviderForProjectId"
    payload = {"projectId": project_id}
    try:
        response = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        response.raise_for_status()
        data = response.json()
        return data.get("data")
    except Exception as e:
        print(f"[ERROR] Could not fetch LLM provider for project {project_id}: {e}")
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
        print(f"[CONFIG] OpenAI API Key loaded (length: {len(openai_key)} chars)")
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
        logger.info("[CONFIG] Fetching Anthropic API key from API endpoint")
        
        response = requests.get(
            "http://localhost:3000/api/secrets/ANTHROPIC_API_KEY",
            headers={'Content-Type': 'application/json'}
        )
        response.raise_for_status()
        data = response.json()
        
        anthropic_key = data['value']
        print(f"[CONFIG] Anthropic API Key loaded (length: {len(anthropic_key)} chars)")
        
        return anthropic_key
        
    except Exception as e:
        print(f"[CONFIG] Failed to fetch Anthropic API key: {e}")
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

def initialize_llm_provider(project_id: int):
    """Initialize LLM provider based on project ID"""
    global llm_provider, client, anthropic_client
    
    provider_name = get_llm_provider_for_project(project_id)
    print(f"[DEBUG] Initializing LLM provider: {provider_name} for project {project_id}")
    
    if provider_name == "OpenAI":
        OPENAI_API_KEY = get_openai_api_key_from_api()
        llm_provider = "openai"
        client = OpenAI(api_key=OPENAI_API_KEY)
        anthropic_client = None
        print(f"[DEBUG] OpenAI client initialized successfully")
    elif provider_name == "Anthropic":
        ANTHROPIC_API_KEY = get_anthropic_api_key_from_api()
        llm_provider = "anthropic"
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        client = None
        print(f"[DEBUG] Anthropic client initialized successfully")
    else:
        raise RuntimeError(f"Unknown provider name: {provider_name}")

# S3 Client Initialization ----------------------------
#AWS_ACCESS_KEY_ID = get_aws_access_key_from_api()
#print("AWS_ACCESS_KEY_ID:", AWS_ACCESS_KEY_ID)
#
#AWS_SECRET_ACCESS_KEY = get_aws_secret_key_from_api()
#print("AWS_SECRET_ACCESS_KEY:", AWS_SECRET_ACCESS_KEY)
#
#AWS_REGION = get_aws_region_from_api()
#print("AWS_REGION:", AWS_REGION)
#
#AWS_BUCKET_NAME = get_aws_bucket_name_from_api()
#print("AWS_BUCKET_NAME:", AWS_BUCKET_NAME)
#
#s3 = boto3.client(
#    's3',
#    aws_access_key_id=AWS_ACCESS_KEY_ID,
#    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#    region_name=AWS_REGION,
#)
# S3 Client Initialization (only if not using local images) ----------------------------
s3 = None
AWS_BUCKET_NAME = None

def initialize_s3_if_needed():
    global s3, AWS_BUCKET_NAME
    if not USE_LOCAL_IMAGES and s3 is None:
        try:
            AWS_ACCESS_KEY_ID = get_aws_access_key_from_api()
            AWS_SECRET_ACCESS_KEY = get_aws_secret_key_from_api()
            AWS_REGION = get_aws_region_from_api()
            AWS_BUCKET_NAME = get_aws_bucket_name_from_api()
            
            s3 = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                region_name=AWS_REGION,
            )
            print("[DEBUG] S3 client initialized")
        except Exception as e:
            print(f"[WARNING] Could not initialize S3: {e}")

def call_llm_unified(messages: list, temperature: float = 0.1) -> str:
    """Unified LLM calling function that uses either OpenAI or Anthropic based on flag"""
    global llm_provider, client, anthropic_client
    
    if llm_provider == "anthropic":
        logger.info("[LLM] Using Anthropic Claude Sonnet 4 model")
        try:
            # Convert OpenAI format to Anthropic format
            if messages[0]["role"] == "system":
                system_msg = messages[0]["content"]
                user_msgs = messages[1:]
            else:
                system_msg = None
                user_msgs = messages
            
            response = anthropic_client.messages.create(
                model="claude-3-7-sonnet-20250219",
                max_tokens=2000,
                temperature=temperature,
                system=system_msg,
                messages=user_msgs
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"[LLM] Anthropic Claude failed: {e}")
            raise
    elif llm_provider == "openai":
        logger.info("[LLM] Using OpenAI GPT-4o model")
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"[LLM] OpenAI failed: {e}")
            raise
    else:
        logger.error(f"[LLM] Unknown provider: {llm_provider}")
        raise RuntimeError(f"Unknown provider: {llm_provider}")


ISO_DT = re.compile(
    r'(?P<prefix>:\s*)'                                    # after a colon
    r'(?P<dt>\d{4}-\d{2}-\d{2}T'                           # YYYY-MM-DDT
    r'\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)'                       # HH:MM:SS(.sss)Z
)

def quote_iso_datetimes(s: str) -> str:
    """Wrap unquoted ISO-8601 datetimes in double quotes."""
    def _repl(m):
        return f'{m.group("prefix")}"{m.group("dt")}"'
    return ISO_DT.sub(_repl, s)

#def parse_api_response(raw: str) -> Dict[str, Any]:
#    """
#    Accepts your raw triple-quoted 'api_response' string (JSON5-like) and returns a dict.
#    json5 tolerates single quotes, trailing commas, unquoted keys, and ISO dates without quotes.
#    """
#    # Trim to first/last brace so we ignore leading/trailing noise in multiline strings.
#    first = raw.find("{")
#    last = raw.rfind("}")
#    if first == -1 or last == -1:
#        raise ValueError("No JSON object found in payload.")
#    core = raw[first:last + 1]
#    core = quote_iso_datetimes(core)    
#    data = json5.loads(core)  # robust to your example payload format
#    return data
def parse_api_response(raw) -> Dict[str, Any]:
    # Handle both dict and string input
    if isinstance(raw, dict):
        return raw
    
    # Handle string input (existing logic)
    first = raw.find("{")
    last = raw.rfind("}")
    if first == -1 or last == -1:
        raise ValueError("No JSON object found in payload.")
    core = raw[first:last + 1]
    core = quote_iso_datetimes(core)    
    data = json5.loads(core)
    return data
#-----local_folder_images-----------------
#def get_all_images_from_folder(folder_path: str = "test_images") -> List[str]:
#    """Get all image files from the specified folder"""
#    if not os.path.exists(folder_path):
#        print(f"Warning: Folder {folder_path} does not exist")
#        return []
#    
#    image_extensions = {'.png', '.jpg', '.jpeg', '.gif'}
#    image_files = []
#    
#    for filename in os.listdir(folder_path):
#        if any(filename.lower().endswith(ext) for ext in image_extensions):
#            image_files.append(os.path.join(folder_path, filename))
#    
#    print(f"Found {len(image_files)} images in {folder_path}: {image_files}")
#    return image_files

# Latest changes ---------------------------------
#def get_all_images_from_folder(folder_path: str = "test_images") -> List[str]:
#    """Get all image files from the specified folder"""
#    print(f"[DEBUG] Scanning folder for images: {folder_path}")
#    
#    if not os.path.exists(folder_path):
#        print(f"[ERROR] Folder {folder_path} does not exist")
#        return []
#    
#    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
#    image_files = []
#    all_files = os.listdir(folder_path)
#    
#    print(f"[DEBUG] Total files in folder: {len(all_files)}")
#    for filename in all_files:
#        print(f"[DEBUG] Checking file: {filename}")
#        if any(filename.lower().endswith(ext) for ext in image_extensions):
#            full_path = os.path.join(folder_path, filename)
#            image_files.append(full_path)
#            print(f"[DEBUG] ✓ Added image: {filename}")
#        else:
#            print(f"[DEBUG] ✗ Skipped non-image: {filename}")
#    
#    print(f"[DEBUG] Final image list ({len(image_files)} files):")
#    for i, img in enumerate(image_files, 1):
#        print(f"[DEBUG]   {i}. {img}")
#    
#    return image_files

from pathlib import Path

def get_all_images_from_folder(folder_path: str = "test_images") -> List[str]:
    """Get all image files from the project root test_images folder."""
    print(f"[DEBUG] Scanning folder for images: {folder_path}")

    # Always resolve relative to the project root (parent of testDataMcp)
    project_root = Path(__file__).resolve().parent.parent
    folder = (project_root / folder_path).resolve()

    if not folder.is_dir():
        print(f"[ERROR] Folder not found: {folder} (cwd={os.getcwd()})")
        return []

    image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
    image_files = [str(f) for f in folder.iterdir() if f.suffix.lower() in image_extensions]

    print(f"[DEBUG] Found {len(image_files)} images in folder: {folder}")
    for i, img in enumerate(image_files, 1):
        print(f"[DEBUG]   {i}. {img}")

    return image_files




#def coerce_steps(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
#    """
#    Ensures we have a flat steps array.
#    Some generators accidentally nest steps: steps: [ steps: [ ... ] ].
#    This flattens that shape if needed.
#    """
#    steps = payload.get("steps")
#    if isinstance(steps, dict) and "steps" in steps:
#        return steps["steps"]
#    return steps or []

def coerce_steps(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Ensures we have a flat steps array.
    Handles both 'steps' and 'stepArray' fields.
    """
    # Try 'steps' first (expected format)
    steps = payload.get("steps")
    if steps:
        if isinstance(steps, dict) and "steps" in steps:
            return steps["steps"]
        return steps
    
    # Fallback to 'stepArray' (from app.js format)
    step_array = payload.get("stepArray")
    if step_array:
        print(f"[DEBUG] Converting stepArray to steps format")
        # Convert stepArray format to steps format
        converted_steps = []
        for i, step in enumerate(step_array, 1):
            converted_step = {
                "tcStepId": i,
                "tcStepDescription": step.get("tcStep", ""),
                "tcStepExpectedResult": step.get("tcResult", "")
            }
            converted_steps.append(converted_step)
        print(f"[DEBUG] Converted {len(converted_steps)} steps")
        return converted_steps
    
    return []





TRANSACTIONAL_CLUES = re.compile(
    r"(\(\.\.\.\d{2,}\)|\b\d{4,}\b|\$\s*\d|[A-Za-z0-9]{8,}|\bID[:#]?\s*\d+)", re.IGNORECASE
)
PLACEHOLDER = re.compile(r"<<[^<>]+>>")

def lint_steps_for_transactional_text(steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flags suspicious hardcoded transactional bits that are not inside placeholders.
    """
    issues = []
    for s in steps:
        for field in ("tcStepDescription", "tcStepExpectedResult"):
            text = s.get(field, "") or ""
            if TRANSACTIONAL_CLUES.search(text) and not PLACEHOLDER.search(text):
                issues.append({
                    "tcStepId": s.get("tcStepId"),
                    "field": field,
                    "snippet": text[:140]
                })
    return issues



STRUCTURED_SCHEMA = {
    "type": "object",
    "additionalProperties": False,  # <-- REQUIRED with strict: true
    "properties": {
        "mode": {"type": "string", "enum": ["patch", "full"]},
        "placeholders_used": {
            "type": "array",
            "items": {"type": "string"},
            "default": []  # optional convenience
        },
        "patched_steps": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,  # <-- add on nested objects too
                "properties": {
                    # allow integer or number (some TCs use ints, some floats)
                    "tcStepId": {"type": ["integer", "number"]},
                    "tcStepDescription": {"type": "string"},
                    "tcStepExpectedResult": {"type": "string"},
                    "reason": {"type": "string"}
                },
                "required": ["tcStepId", "tcStepDescription", "tcStepExpectedResult","reason"]
            }
        }
    },
    "required": ["mode","placeholders_used", "patched_steps"]
}


def path_to_data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

#def images_to_content_blocks(image_paths: List[str]) -> List[dict]:
#    blocks = []
#    for p in image_paths:
#        blocks.append({
#            "type": "image_url",
#            "image_url": { "url": path_to_data_url(p) }   # <-- object with url
#        })
#    return blocks



# SYSTEM_PROMPT = """You are a Test Case Normalizer.

# ## Objective
# Rewrite test steps so that **no transactional value ever appears as a hardcoded literal**. Replace all such values with placeholders. Keep only true master (static) values as literals. Preserve each step’s intent and `tcStepId`.

# ## Non-negotiable Rules
# 1) If a step contains any transactional value, you MUST:
#    - Replace it with a placeholder in the form `<<UPPER_SNAKE_CASE>>`.
#    - Include that step in `patched_steps` with a short `reason`.
#    - Do not skip. Do not leave any transactional literal.

# 2) Master/static values MUST remain literal. Examples: page titles, static labels, static dropdown for all users, section headers, menu/button captions, fixed feature names that do not vary per user/session.
# Strictly do not replace with placeholders for master values.
# 3) Do NOT add, delete, split, or reorder steps. Only correct fields of existing steps.

# 4) Placeholders:
#    - Format: `<<UPPER_SNAKE_CASE>>` (letters, digits, underscores only).
#    - Derive the placeholder name from the **field label** or **domain meaning** (e.g., Email Address → <<EMAIL_ADDRESS>>).
#    - Reuse an existing placeholder if it already represents the same concept.

# ## What counts as “Transactional” (MUST be placeholders)
# - Any value that can vary by user/session/transaction, including:
#   - Account numbers, masked accounts
#   - IDs / reference numbers
#   - Emails, phone numbers, user names, addresses
#   - Monetary amounts, balances, quantities, exchange rates
#   - Dates/times/timestamps, confirmation codes, OTPs
#   - Any quoted option that is user-specific (e.g., an account nickname or dynamically generated label)
# - Dropdowns with prefilled or user-specific options:
#   - Do not name a concrete option; replace it with a placeholder derived from the field label.
#   - You MUST always use a placeholder only for transaction values

# ## Output Contract
# - Return JSON that matches the provided schema EXACTLY.
# - PATCH mode: include ONLY changed steps in `patched_steps`.
# - FULL mode: include EVERY step (already corrected) in `patched_steps`.
# - Always include a brief `reason` for each patched step explaining the change.

# Be precise, consistent, and exhaustive in removing transactional literals. If any transactional literal remains, you must patch it."""

SYSTEM_PROMPT = """You are a Test Case Normalizer.

## Objective
-Rewrite test steps so that **no transactional value** appears as a hardcoded literal. Replace transactionals with placeholders. Keep **master/static** values literal. Preserve each step’s intent and `tcStepId`. 
- Any transactional literal  MUST be replaced with the same placeholder **everywhere it occurs**: in descriptions, expected results, verification steps, assertions, and messages.
- Do not assume that transactional literals in verification/assertion steps are master values. They are transactional and MUST be replaced consistently across all steps.


## Silent reasoning (do not output this)
1) From the provided screenshot(s), **identify form controls** and their labels.
2) **Classify each control internally**:
   - **master_enum**: a global, fixed list (e.g., Status: New, Pending Review, Info Required, Approved, Rejected).
   - **transactional_enum**: user/session-specific list or selected value showing PII or money (e.g., “From Account: Checking (...4455) – $14,550.75”).
3) Use this internal map to decide what to keep literal vs placeholder.

## Decision rules
A. **Master values → keep literal** (use exact canonical spelling seen in the UI). Master values are the one which is same across all the users, it wont differ from one user to another. Values that no need test data to be generated seperately.
   - Examples: page/section headers, button labels, fixed menu items, global dropdown options that are same for all other users.
   
B. **Transactional values → placeholder**:
   - Values that depend on the user which need test data to be generated unlike master values. Anything user/session-specific or sensitive: emails, names, phone numbers, addresses; balances and **any currency amounts**; dates/times; confirmation/OTP codes;
    
Some dropdowns have options that might be master values and some dropdowns are transaction values which change from one user to another. You should analyze the test case and the image then distinguish them properly.
## Placeholder specification
- Format: `<<UPPER_SNAKE_CASE>>`.
- Naming algorithm:
  1) Start from the field/control label or closest domain concept.
  2) Uppercase; replace non-alphanumerics with `_`; collapse repeats; trim `_`.
  3) Reuse the same placeholder for the same concept across steps.

  ## Examples (generic, not tied to any specific app)
- “Set Role to ‘Editor’” → keep **Editor** literal (**master_enum**).
- “Enter ‘555-0199’ in Phone” → `<<PHONE_NUMBER>>`.
- “Verify order reference ‘ORD-88992’ is displayed” → `<<ORDER_REFERENCE>>` (keep label **Order reference** literal).
- “Choose Severity ‘High’” → keep **High** literal (**master_enum**).

## Non-negotiable constraints
1) Do not add, delete, split, or reorder steps—only adjust text of existing steps.
2) Every transactional change must appear in `patched_steps` with a brief `reason` (≤10 words).
3) Do not invent new controls/placeholders beyond what is needed.
4) Do not output your reasoning or the internal control map.

## Output contract (STRICT)
- Return JSON that matches the provided schema exactly.
- PATCH mode: only changed steps in `patched_steps`.
- FULL mode: every step.
- Also include `placeholders_used` listing unique placeholders you inserted.

Be precise and exhaustive: if any transactional literal remains, you must patch it."""



def normalize_with_openai_multimodal(
    steps: List[Dict[str, Any]],
    image_paths: List[str],
    mode: str = "patch",
    project_id:int = None
) -> Dict[str, Any]:
    """
    Sends steps + page screenshots (one or many) in one Responses API call.
    Returns: { mode, patched_steps: [...] } per the schema.
    """
    #client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    provider_name = get_llm_provider_for_project(project_id)
    
    if provider_name == "OpenAI":
        client = OpenAI(api_key=get_openai_api_key_from_api())
        model = "gpt-4o-mini"
    elif provider_name == "Anthropic":
        # Handle Anthropic client initialization
        # Note: You'll need to adapt this for Anthropic's multimodal capabilities
        raise NotImplementedError("Anthropic multimodal support needs implementation")
    
    if not image_paths:
        raise ValueError("Provide at least one local image path (.png/.jpg).")

    user_content = [
        
        {"type": "text",
        "text": (
            "Return a JSON " + mode.upper() + " correction. "
            "Include only changed steps in 'patched_steps' for PATCH; "
            "include every step for FULL (already corrected). "
            "Here are the steps:\n" + json.dumps(steps, ensure_ascii=False)
        ),},
        *images_to_content_blocks(image_paths, from_s3=False)
    ]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content}
    ]
    try:
        chat = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "NormalizedPatch",
                    "schema": STRUCTURED_SCHEMA,
                    "strict": True
                }
            },
            temperature=0
        )
        out_text = chat.choices[0].message.content
        return json.loads(out_text)
# ...existing code...
#def normalize_with_openai_multimodal(
#    steps: List[Dict[str, Any]],
#    image_paths: List[str],
#    mode: str = "patch",
#    model: str = "gpt-4o-mini"
#) -> Dict[str, Any]:
#    """
#    Sends steps + page screenshots (one or many) in one Responses API call.
#    Returns: { mode, patched_steps: [...] } per the schema.
#    """
#    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#
#    # Remove this check for development/testing without images:
#    # if not image_paths:
#    #     raise ValueError("Provide at least one local image path (.png/.jpg).")
#
#    user_content = [
#        {"type": "text",
#        "text": (
#            "Return a JSON " + mode.upper() + " correction. "
#            "Include only changed steps in 'patched_steps' for PATCH; "
#            "include every step for FULL (already corrected). "
#            "Here are the steps:\n" + json.dumps(steps, ensure_ascii=False)
#        ),},
#        *images_to_content_blocks(image_paths) if image_paths else []
#    ]
#    messages = [
#        {"role": "system", "content": SYSTEM_PROMPT},
#        {"role": "user", "content": user_content}
#    ]
#    try:
#        chat = client.chat.completions.create(
#            model=model,
#            messages=messages,
#            response_format={
#                "type": "json_schema",
#                "json_schema": {
#                    "name": "NormalizedPatch",
#                    "schema": STRUCTURED_SCHEMA,
#                    "strict": True
#                }
#            },
#            temperature=0
#        )
#        out_text = chat.choices[0].message.content
#        return json.loads(out_text)

    except TypeError:
        # Fallback to JSON object mode if json_schema isn't supported in your env:
        chat = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0
        )
        out_text = chat.choices[0].message.content
        return json.loads(out_text)
# ...existing code...
    except TypeError:
        # Fallback to JSON object mode if json_schema isn't supported in your env:
        chat = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=0
        )
        out_text = chat.choices[0].message.content
        return json.loads(out_text)

# ---------------S3 storage image information--------------------
def s3_image_to_data_url(s3_key: str) -> str:
    """Convert S3 image to data URL"""
    try:
        response = s3.get_object(Bucket=AWS_BUCKET_NAME, Key=s3_key)
        image_data = response['Body'].read()
        
        # Guess MIME type from file extension
        mime = mimetypes.guess_type(s3_key)[0] or "image/png"
        b64 = base64.b64encode(image_data).decode("utf-8")
        return f"data:{mime};base64,{b64}"
    except Exception as e:
        print(f"Error reading image from S3: {e}")
        raise

def images_to_content_blocks(image_paths: List[str], from_s3: bool = True) -> List[dict]:
    """Updated to handle both local and S3 images"""
    blocks = []
    for path in image_paths:
        if from_s3:
            data_url = s3_image_to_data_url(path)
        else:
            data_url = path_to_data_url(path)  # existing local function
        
        blocks.append({
            "type": "image_url",
            "image_url": {"url": data_url}
        })
    return blocks
# ---------------------------
# 4) Merge logic
# ---------------------------

def merge_patched_steps(original_steps: List[Dict[str, Any]],
                        patched_steps: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    For each object in patched_steps, replace description/expected in the step
    that has the same tcStepId.
    """
    by_id = {s["tcStepId"]: s for s in original_steps}
    for p in patched_steps:
        sid = p["tcStepId"]
        if sid in by_id:
            by_id[sid] = {
                **by_id[sid],
                "tcStepDescription": p["tcStepDescription"],
                "tcStepExpectedResult": p["tcStepExpectedResult"],
            }
    # Return in the original order:
    return [by_id[s["tcStepId"]] for s in original_steps]


# ---------------------------
# 5) Main entry
# ---------------------------

#def normalize_payload(
#    raw: str,
#    image_paths: List[str],
#    project_id: int, 
#    mode: str = "patch",
#    model: str = "gpt-4o-mini"
#) -> Dict[str, Any]:
#    """
#    - Parses your input payload string,
#    - Calls OpenAI with steps + one/many screenshots (PNG/JPG),
#    - Merges corrections if any,
#    - Returns full payload dict (unchanged if nothing needed fixing).
#    """
#    payload = parse_api_response(raw)
#    steps = coerce_steps(payload)
#
#    # Optional pre-check; you can ignore the result if you like.
#    _ = lint_steps_for_transactional_text(steps)
#
#    result = normalize_with_openai_multimodal(steps, image_paths, mode=mode, project_id=project_id)
#    patched = result.get("patched_steps", [])
#
#    if not patched:        # no issues found -> return as-is
#        return payload
#
#    payload["steps"] = merge_patched_steps(steps, patched)
#    return payload

def normalize_payload(
    raw,  # Can be dict or str
    image_folder: str = "test_images",  # Folder path instead of specific images
    mode: str = "patch"
) -> Dict[str, Any]:
    """
    - Parses input payload (dict or string),
    - Gets all images from specified folder,
    - Calls LLM with steps + all screenshots,
    - Returns cleaned payload dict.
    """
    payload = parse_api_response(raw)

    print(f"[DEBUG] Test case structure: {list(payload.keys())}")
    print(f"[DEBUG] Steps field exists: {'steps' in payload}")
    if 'stepArray' in payload:
        print(f"[DEBUG] Found stepArray instead of steps")

    #Extracting project_id from payload
    project_id = payload.get("projectId")
    if not project_id:
        raise ValueError("projectId not found in test case data")
    print(f"[DEBUG] Extracted project ID: {project_id} from test case")

    steps = coerce_steps(payload)
    print(f"[DEBUG] Extracted {len(steps)} steps from test case")
    # Get all images from folder
    image_paths = get_all_images_from_folder(image_folder)
    print(f"[DEBUG] Found {len(image_paths)} images in folder: {image_folder}")
    for i, img_path in enumerate(image_paths, 1):
        print(f"[DEBUG] Image {i}: {img_path}")
    
    if not image_paths:
        print("Warning: No images found, returning payload unchanged")
        return payload

    # Initialize LLM provider
    initialize_llm_provider(project_id)

    result = normalize_with_openai_multimodal(steps, image_paths, mode=mode, project_id=project_id)
    patched = result.get("patched_steps", [])

    if not patched:
        print("[DEBUG] No patches needed, returning original payload")
        return payload

    #payload["steps"] = merge_patched_steps(steps, patched)
    #print(f"[DEBUG] Applied {len(patched)} patches to test case")
    #print(f"[DEBUG] Final cleaned test case:")
    #print(json.dumps(payload, indent=2, ensure_ascii=False))
    #return payload

    if "stepArray" in payload and "steps" not in payload:
  
        cleaned_stepArray = []
        final_steps = merge_patched_steps(steps, patched)
        for step in final_steps:
            cleaned_stepArray.append({
                "tcStep": step.get("tcStepDescription", ""),
                "tcResult": step.get("tcStepExpectedResult", "")
            })
        payload["stepArray"] = cleaned_stepArray
        print(f"[DEBUG] Converted back to stepArray format with {len(cleaned_stepArray)} steps")
    else:
        payload["steps"] = merge_patched_steps(steps, patched)
    
    print(f"[DEBUG] Applied {len(patched)} patches to test case")
    print(f"[DEBUG] Final cleaned test case:")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload

# ---------------------------
# 6) Example usage (if run as script)
# ---------------------------

if __name__ == "__main__":
    print("Cleaning agent ready. Use normalize_payload() function to clean test cases.")
    print("Example usage:")
    print("result = normalize_payload(test_case_dict, image_folder='test_images', mode='patch')")

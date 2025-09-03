
// src/mcp_server.ts
// To run:
// 1. npm init -y
// 2. npm install ws @playwright/test uuid @types/ws @types/node
// 3. npx tsc --init
// 4. In tsconfig.json: set
//     "target": "es2020",
//     "module": "commonjs",
//     "outDir": "./dist",
//     "rootDir": "./src"
// 5. Add to package.json scripts:
//     "start:mcp": "node dist/mcp_server.js"
// 6. npm run start:mcp

import { WebSocketServer } from 'ws';
import { chromium, Page, Browser, Locator,expect } from '@playwright/test';
import { v4 as uuidv4 } from 'uuid';
import * as http from 'http';
import { inspect } from 'util';
import fs from 'fs/promises';
import path from 'path';
import dotenv from 'dotenv';
import jwt from 'jsonwebtoken';
import crypto from 'crypto';
import url from 'url';
import * as mysql from 'mysql2/promise';

let wss: WebSocketServer | null = null;
const JWT_SECRET = crypto.randomBytes(64).toString('hex'); // Generate random secret on startup


let VALID_AGENT_ID: string;
let VALID_AGENT_SECRET: string;

let PERSISTENT_JWT_TOKEN: string = '';
const JWT_REFRESH_INTERVAL_MINUTES = 15;//refresh time
let PREVIOUS_JWT_TOKEN: string = '';
let tokenRefreshTimer: NodeJS.Timeout | null = null;

function initializePersistentToken() {
  generateNewJWTToken();
  startTokenRefreshTimer();
  console.log(`[JWT] Token refresh system initialized - refreshing every ${JWT_REFRESH_INTERVAL_MINUTES} minutes`);
}

function generateNewJWTToken() {
  const payload = {
    agentId: VALID_AGENT_ID,
    serverName: 'CustomPlaywrightMCP',
    version: '1.0.0',
    generatedAt: new Date().toISOString(),
    purpose: 'authenticated_agent_access',
    sessionToken: true,
    tokenId: crypto.randomBytes(8).toString('hex')
  };
  
  PREVIOUS_JWT_TOKEN = PERSISTENT_JWT_TOKEN;
  PERSISTENT_JWT_TOKEN = jwt.sign(payload, JWT_SECRET, { expiresIn: `${JWT_REFRESH_INTERVAL_MINUTES + 5}m` });
  
  const currentTime = new Date().toISOString();
  console.log(`[JWT] New token generated at ${currentTime}`);
  console.log(`[JWT] Token expires in: ${JWT_REFRESH_INTERVAL_MINUTES + 5} minutes`);
  console.log(`[JWT] New JWT Token: ${PERSISTENT_JWT_TOKEN}`);
  broadcastTokenRefresh();
}

function startTokenRefreshTimer() {
  if (tokenRefreshTimer) {
    clearInterval(tokenRefreshTimer);
  }
  
  tokenRefreshTimer = setInterval(() => {
    console.log('[JWT] Generating new JWT token (scheduled refresh)');
    generateNewJWTToken();
  }, JWT_REFRESH_INTERVAL_MINUTES * 60 * 1000);
}

function broadcastTokenRefresh() {
  if (wss && wss.clients) { // Changed from typeof check to null check
    wss.clients.forEach((client: any) => { // Added type annotation
      if (client.readyState === 1) {
        try {
          client.send(JSON.stringify({
            jsonrpc: '2.0',
            method: 'token_refresh',
            params: {
              newToken: PERSISTENT_JWT_TOKEN,
              expiresIn: `${JWT_REFRESH_INTERVAL_MINUTES}m`,
              refreshedAt: new Date().toISOString()
            }
          }));
          console.log('[JWT] Token refresh broadcasted to connected client');
        } catch (error) {
          console.error('[JWT] Failed to broadcast token refresh:', error);
        }
      }
    });
  } else {
    console.log('[JWT] WebSocket server not available for token broadcast');
  }
}

async function retry<T>(
  operation: () => Promise<T>,
  maxRetries: number = 10,
  delayMs: number = 1000, // 1 second delay
  description: string = 'operation'
): Promise<T> {
  let attempt = 0;
  while (attempt < maxRetries) {
    try {
      console.log(`[RETRY] Attempt ${attempt + 1}/${maxRetries} for ${description}...`);
      return await operation();
    } catch (error: any) {
      console.warn(`[RETRY] ${description} failed on attempt ${attempt + 1}: ${error.message}`);
      attempt++;
      if (attempt < maxRetries) {
        console.log(`[RETRY] Retrying ${description} in ${delayMs / 1000} seconds...`);
        await new Promise(resolve => setTimeout(resolve, delayMs));
      }
    }
  }
  throw new Error(`[RETRY] All ${maxRetries} attempts failed for ${description}.`);
}

async function getCredentialsFromAPI(): Promise<{ agentId: string; secretKey: string }> {
  const fetchSecrets = async () => {
    console.debug('[API] Loading credentials from API endpoint...');

    // Fetch agent ID
    const agentIdResponse = await fetch('http://localhost:3000/api/secrets/AGENT_ID', {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!agentIdResponse.ok) {
      // Throwing an error here will trigger a retry
      throw new Error(`Agent ID fetch failed: ${agentIdResponse.status} - ${await agentIdResponse.text()}`);
    }

    const agentIdData = await agentIdResponse.json();
    console.debug('[API] Agent ID response:', agentIdData);

    // Fetch secret key
    const secretKeyResponse = await fetch('http://localhost:3000/api/secrets/AGENT_SECRET_KEY', {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });

    if (!secretKeyResponse.ok) {
      // Throwing an error here will trigger a retry
      throw new Error(`Secret key fetch failed: ${secretKeyResponse.status} - ${await secretKeyResponse.text()}`);
    }

    const secretKeyData = await secretKeyResponse.json();
    console.debug('[API] Secret key response:', secretKeyData);

    const agentId = agentIdData.value;
    const secretKey = secretKeyData.value;

    if (!agentId || !secretKey) {
      // Throwing an error here will trigger a retry
      throw new Error('Agent ID or Secret Key not found in API response');
    }

    console.debug('[API] Credentials loaded from API successfully');
    console.debug('[API] Agent ID:', agentId);
    console.debug('[API] Secret key length:', secretKey.length);

    return { agentId, secretKey };
  };

  try {
    // Retry fetching the secrets up to 15 times with a 2-second delay
    return await retry(fetchSecrets, 15, 2000, 'Agent credentials');
  } catch (error) {
    console.error('[API] Final failure to load credentials from API:', error);
    throw error; // Re-throw the error after all retries fail
  }
}

// Store active JWT tokens per agent
//const activeTokens: Map<string, string> = new Map();

// Verify static credentials
function verifyStaticCredentials(agentId: string, secretKey: string): boolean {
  console.debug('[AUTH]  Validating static credentials...');
  console.debug('[AUTH]  Expected Agent ID:', VALID_AGENT_ID);
  console.debug('[AUTH]  Received Agent ID:', agentId);
  console.debug('[AUTH]  Secret key length expected:', VALID_AGENT_SECRET?.length || 0);
  console.debug('[AUTH]  Secret key length received:', secretKey.length);
  
  const isValid = agentId === VALID_AGENT_ID && secretKey === VALID_AGENT_SECRET;
  console.debug('[AUTH]  Static credentials valid:', isValid);
  return isValid;
}

// Generate JWT token for authenticated agent
function generateJWTToken(agentId: string): string {
  const payload = {
    agentId: agentId,
    serverName: 'CustomPlaywrightMCP',
    version: '1.0.0',
    generatedAt: new Date().toISOString(),
    purpose: 'authenticated_agent_access'
  };
  
  if (!PERSISTENT_JWT_TOKEN) {
    throw new Error('Persistent token not initialized');
  }
  console.debug('[JWT] Returning persistent token to agent:', agentId);
  return PERSISTENT_JWT_TOKEN;
}

function verifyJWTToken(token: string): { valid: boolean; agentId?: string; needsRefresh?: boolean } {
  try {
    if (token === PERSISTENT_JWT_TOKEN) {
      const decoded = jwt.verify(token, JWT_SECRET) as any;
      console.debug('[JWT] Current token validated for agent:', decoded.agentId);
      return { valid: true, agentId: decoded.agentId };
    }
    
    if (token === PREVIOUS_JWT_TOKEN && PREVIOUS_JWT_TOKEN) {
      const decoded = jwt.verify(token, JWT_SECRET) as any;
      console.debug('[JWT] Previous token accepted (grace period) for agent:', decoded.agentId);
      return { valid: true, agentId: decoded.agentId, needsRefresh: true };
    }
    
    console.debug('[JWT] Token does not match current or previous session token');
    return { valid: false };
  } catch (error) {
    console.error('[JWT] Token verification failed:', error);
    return { valid: false };
  }
}
// Verify JWT token
//function verifyJWTToken(token: string): { valid: boolean; agentId?: string } {
//  try {
//    const decoded = jwt.verify(token, JWT_SECRET) as any;
//    const agentId = decoded.agentId;
//    const storedToken = activeTokens.get(agentId);
//    
//    console.debug('[JWT]  Validating JWT token for agent:', agentId);
//    console.debug('[JWT]  Token signature valid:', true);
//    console.debug('[JWT]  Token matches stored token:', token === storedToken);
//    
//    if (token === storedToken) {
//      return { valid: true, agentId };
//    } else {
//      console.debug('[JWT]  Token does not match stored token');
//      return { valid: false };
//    }
//  } catch (error) {
//    console.error('[JWT]  Token verification failed:', error);
//    return { valid: false };
//  }
//}

// Display server authentication info
// Display server authentication info
function displayServerInfo() {
  console.log('\n' + '='.repeat(80));
  console.log(' MCP SERVER AUTHENTICATION INFO');
  console.log('='.repeat(80));
  console.log(' Authentication Flow:');
  console.log('   1. Agent sends Agent ID + Secret Key for initial auth');
  console.log('   2. Server validates credentials and generates JWT token');
  console.log('   3. Agent uses JWT token for all subsequent requests');
  console.log('\n Required Agent Credentials:');
  console.log('   Agent ID:', VALID_AGENT_ID || 'NOT SET');
  console.log('   Secret Key Length:', VALID_AGENT_SECRET?.length || 0, 'characters');
  console.log('\n Connection URL:');
  console.log(`   ws://localhost:${MCP_PORT}`);
  console.log('='.repeat(80) + '\n');
}

async function getRDSCredentialsFromAPI(): Promise<{
  rdsHost: string;
  rdsUser: string;
  rdsPass: string;
  rdsDb: string;
}> {
  const fetchRDSSpecs = async () => {
    console.debug('[API] Loading RDS credentials from API endpoint...');

    // Fetch RDS credentials in parallel
    const [rdsHostResponse, rdsUserResponse, rdsPassResponse, rdsDbResponse] = await Promise.all([
      fetch('http://localhost:3000/api/secrets/RDS_HOST', { method: 'GET', headers: { 'Content-Type': 'application/json' } }),
      fetch('http://localhost:3000/api/secrets/RDS_USER', { method: 'GET', headers: { 'Content-Type': 'application/json' } }),
      fetch('http://localhost:3000/api/secrets/RDS_PASS', { method: 'GET', headers: { 'Content-Type': 'application/json' } }),
      fetch('http://localhost:3000/api/secrets/RDS_DB', { method: 'GET', headers: { 'Content-Type': 'application/json' } })
    ]);

    // Check responses (throw if not ok, so retry is triggered)
    if (!rdsHostResponse.ok) throw new Error(`RDS_HOST fetch failed: ${rdsHostResponse.status} - ${await rdsHostResponse.text()}`);
    if (!rdsUserResponse.ok) throw new Error(`RDS_USER fetch failed: ${rdsUserResponse.status} - ${await rdsUserResponse.text()}`);
    if (!rdsPassResponse.ok) throw new Error(`RDS_PASS fetch failed: ${rdsPassResponse.status} - ${await rdsPassResponse.text()}`);
    if (!rdsDbResponse.ok) throw new Error(`RDS_DB fetch failed: ${rdsDbResponse.status} - ${await rdsDbResponse.text()}`);

    // Parse responses
    const [rdsHostData, rdsUserData, rdsPassData, rdsDbData] = await Promise.all([
      rdsHostResponse.json(),
      rdsUserResponse.json(),
      rdsPassResponse.json(),
      rdsDbResponse.json()
    ]);

    const credentials = {
      rdsHost: rdsHostData.value,
      rdsUser: rdsUserData.value,
      rdsPass: rdsPassData.value,
      rdsDb: rdsDbData.value
    };

    if (!credentials.rdsHost || !credentials.rdsUser || !credentials.rdsPass || !credentials.rdsDb) {
      throw new Error('Some RDS credentials not found in API response');
    }

    console.debug('[API] RDS credentials loaded from API successfully');
    console.debug('[API] RDS Host:', credentials.rdsHost);
    console.debug('[API] RDS User:', credentials.rdsUser);
    console.debug('[API] RDS Database:', credentials.rdsDb);
    console.debug('[API] RDS Password length:', credentials.rdsPass.length, 'characters');

    return credentials;
  };

  try {
    // Retry fetching RDS credentials up to 15 times with a 2-second delay
    return await retry(fetchRDSSpecs, 15, 2000, 'RDS credentials');
  } catch (error) {
    console.error('[API] Final failure to load RDS credentials from API:', error);
    throw error; // Re-throw the error after all retries fail
  }
}
// --- JSON-RPC Request/Response Types ---
interface JsonRpcRequest {
  jsonrpc: '2.0';
  id: number;
  method: string;
  params?: any;
}
interface JsonRpcResponse {
  jsonrpc: '2.0';
  id: number;
  result?: any;
  error?: { code: number; message: string; data?: any };
}

// --- BrowserTool Context & Definition ---
interface BrowserToolContext {
  page: Page;
  browser: Browser;
  session_id: string;
}
interface Tool {
  name: string;
  description: string;
  parameters: {
    type: 'object';
    properties: { [key: string]: any };
    required?: string[];
  };
  execute: (ctx: BrowserToolContext, args: any) => Promise<any>;
}

// --- Helper to safely get element text ---
async function getElementText(locator: Locator, maxLength = 100): Promise<string> {
  try {
    const t = (await locator.textContent())?.trim() || (await locator.innerText())?.trim() || '';
    return t.substring(0, maxLength);
  } catch {
    return '';
  }
}

// --- Tool Implementations ---

const browserNavigateTool: Tool = {
  name: 'browser_navigate',
  description: 'Navigate to a URL',
  parameters: {
    type: 'object',
    properties: {
      url: { type: 'string', description: 'The URL to navigate to.' }
    },
    required: ['url']
  },
  async execute(ctx, args) {
    await ctx.page.goto(args.url, { waitUntil: 'domcontentloaded',timeout: 60000  });
    return {
      success: true,
      message: `Navigated to ${args.url}`,
      pageUrl: ctx.page.url(),
      pageTitle: await ctx.page.title()
    };
  }
};

const browserTypeTool: Tool = {
  name: 'browser_type',
  description: 'Type text into an element (tries multiple selectors)',
  parameters: {
    type: 'object',
    properties: {
      selectors: {
        type: 'array',
        items: { type: 'string' },
        description: 'Array of Playwright locator expressions to try'
      },
      text: { type: 'string', description: 'Text to type' }
    },
    required: ['selectors', 'text']
  },
  async execute(ctx, args) {
    const { selectors, text } = args as { selectors: string[]; text: string };
    let lastError: any;
    for (const sel of selectors) {
      try {
        // e.g. sel == "locator('#txtUsername')" or "getByLabel('Name')"
        const locator: Locator = await eval(`ctx.page.${sel}`);
        await locator.fill(text);
        return { success: true, message: `Typed "${text}" into ${sel}`, succeededSelector:sel };
      } catch (e: any) {
        lastError = e;
        console.warn(`Type failed for ${sel}: ${e.message}`);
      }
    }
    throw new Error(
      `All type selectors failed: ${JSON.stringify(selectors)}\n` +
      `Last error: ${lastError?.message || lastError}`
    );
  }
};


function normalizeSelector(input: string): string {
  // trim + strip surrounding backticks/quotes
  let s = input.trim().replace(/^[`'"]|[`'"]$/g, "");

  // strip optional "page." prefix
  if (s.startsWith("page.")) s = s.slice(5);

  // locator('...') -> inner
  let m = s.match(/^locator\((['"`])(.*)\1\)\s*$/);
  if (m) return asCssOrEngine(m[2]);

  // getByText('...') -> text=...
  m = s.match(/^getByText\((['"`])(.*)\1\)\s*$/);
  if (m) return `text=${m[2]}`;

  // getByRole('role', { name: '...' }) -> role=role[name="..."]
  m = s.match(/^getByRole\((['"`])([\w-]+)\1\s*,\s*\{\s*name\s*:\s*(['"`])(.*)\3\s*\}\s*\)\s*$/);
  if (m) return `role=${m[2]}[name="${m[4]}"]`;

  // Already looks like an engine selector (css=, text=, role=, xpath=, id=, data-testid=)
  if (/^(css|text|role|xpath|id|data-testid)=/i.test(s)) return s;

  // Fallback: treat as raw CSS
  return asCssOrEngine(s);
}

// Helpers

function asCssOrEngine(sel: string): string {
  // If it looks like an XPath, use xpath=…
  if (/^\/\//.test(sel) || /^\(.+\)$/.test(sel)) return `xpath=${sel}`;
  // Otherwise assume CSS; auto-escape common Tailwind chars
  return `css=${escapeTailwind(sel)}`;
}

function escapeTailwind(css: string): string {
  // Escape variant ":" and slash "/" inside class names
  return css
    .replace(/(\.[\w-]+):/g, "$1\\:")   // .hover:bg-... -> .hover\:bg-...
    .replace(/(\.[\w-]+)\//g, "$1\\/"); // .w-1/2 -> .w-1\/2
}



const browserClickTool: Tool = {
  name: 'browser_click',
  description: 'Click an element, trying multiple selectors in order',
  parameters: {
    type: 'object',
    properties: {
      selectors: { type: 'array', items: { type: 'string' } }
    },
    required: ['selectors']
  },
  async execute(ctx, args) {
    const raw = args.selectors as string[];
    let lastError: any;

    for (const original of raw) {
      const sel = normalizeSelector(original);
      try {
        const loc = ctx.page.locator(sel).first();
        await loc.waitFor({ state: 'visible', timeout: 5000 });
        await loc.click();
        return { success: true, message: `Clicked using "${sel}"`, succeededSelector: original };
      } catch (e: any) {
        lastError = e;
        console.warn(`Click failed for "${sel}": ${e.message}`);
      }
    }
    throw new Error(
      `All click selectors failed: ${JSON.stringify(raw)}\nLast error: ${lastError?.message || lastError}`
    );
  }
};






export const optionSelectTool: Tool = {
  name: 'option_select',
  description: 'Handles selecting an option from both native HTML <select> dropdowns and React-Select components by inferring identifier from selector, with enhanced robustness.',
  parameters: {
    type: 'object',
    properties: {
      selector: { // Expects the Playwright selector string directly from the client
        type: 'string',
        description: 'The Playwright selector string for the dropdown element (e.g., "select#id", "page.getByLabel(\'Label\')", "page.locator(\'#react-select-xyz\')").',
      },
      value: { // Renamed from optionText to match client's 'value'
        type: 'string',
        description: 'The exact visible text of the option to be selected.',
      },
    },
    required: ['selector', 'value'],
  },
  /**
   * Executes the dropdown selection logic.
   * @param ctx Context object containing the Playwright page instance.
   * @param args Arguments passed to the tool, including selector and value.
   * @returns A success message or an error message.
   */
  async execute(ctx: { page: Page }, args: { selector: string; value: string }): Promise<{ success: boolean; message: string }> {
    const { page } = ctx;
    const clientSelector = args.selector;
    const optionText = args.value;

    console.log(`[option_select] Attempting to select '${optionText}' using client selector '${clientSelector}'...`);

    let dropdownIdentifier: string = '';

    // --- Refined Inference for dropdownIdentifier (remains similar, focusing on clean strings) ---
    const labelMatch = clientSelector.match(/getByLabel\(['"](.*?)['"]\)/);
    if (labelMatch) {
      dropdownIdentifier = labelMatch[1];
    } else {
      const roleNameMatch = clientSelector.match(/getByRole\('combobox', \{ name: ['"](.*?)['"] \}\)/);
      if (roleNameMatch) {
        dropdownIdentifier = roleNameMatch[1];
      } else {
        const idMatch = clientSelector.match(/(?:select|input|div|span)#([\w-]+)/);
        if (idMatch) {
          dropdownIdentifier = idMatch[1];
        } else {
          const textLocatorMatch = clientSelector.match(/getByText\(['"](.*?)['"]\)/);
          if (textLocatorMatch) {
              dropdownIdentifier = textLocatorMatch[1];
          }
        }
      }
    }
    const displayIdentifier = dropdownIdentifier || clientSelector;
    console.log(`[option_select] Inferred clean identifier: '${dropdownIdentifier}' (Display used for logs: '${displayIdentifier}').`);


    // --- Strategy 1: Attempt to handle Native HTML <select> dropdown ---
    function resolveLocator(page: Page, clientSelector: string): Locator {
      const s = clientSelector.trim();

      // Accept "page.locator('...')" and friends
      const pageCall = s.match(/^page\.(getByLabel|getByRole|getByText|locator)\(([\s\S]*)\)$/);
      if (pageCall) {
        const [, fn, args] = pageCall;
        // Very small parser for common cases; args is the inside of the parentheses.
        // We’ll support the simple, common forms used by your client.
        try {
          if (fn === 'locator') {
            // page.locator('css') or page.locator(`css`)
            const m = args.match(/^[`'"]([\s\S]*?)[`'"]\s*\)$/) || args.match(/^[`'"]([\s\S]*?)[`'"]\s*$/);
            if (m) return page.locator(m[1]);
          } else if (fn === 'getByText') {
            const m = args.match(/^[`'"]([\s\S]*?)[`'"]\s*(,\s*\{[^}]*\})?\s*$/);
            if (m) return page.getByText(m[1]);
          } else if (fn === 'getByLabel') {
            const m = args.match(/^[`'"]([\s\S]*?)[`'"]\s*$/);
            if (m) return page.getByLabel(m[1]);
          } else if (fn === 'getByRole') {
            // getByRole('combobox', { name: 'From Account' })
            const nameMatch = args.match(/^\s*[`'"](\w+)[`'"]\s*,\s*\{\s*name:\s*[`'"]([\s\S]*?)[`'"]\s*\}\s*$/);
            if (nameMatch) return page.getByRole(nameMatch[1] as any, { name: nameMatch[2] });
            const simpleMatch = args.match(/^\s*[`'"](\w+)[`'"]\s*$/);
            if (simpleMatch) return page.getByRole(simpleMatch[1] as any);
          }
        } catch { /* fall through */ }
      }

      // Accept "locator('...')" (no "page.")
      const bareLoc = s.match(/^locator\(([\s\S]*)\)$/);
      if (bareLoc) {
        const m = bareLoc[1].match(/^[`'"]([\s\S]*?)[`'"]\s*$/);
        if (m) return page.locator(m[1]);
      }

      // As a last resort, treat as a direct selector
      return page.locator(s);
    }

    // --- Strategy 1: robust Native <select> handling (value-first, then label, then heuristics) ---
    console.log(`[option_select] Strategy 1: Checking for native HTML <select> for '${displayIdentifier}'...`);
    try {
      let nativeSelectLocator: Locator = resolveLocator(page, clientSelector);

      // If the given selector isn't a SELECT, try to infer by id/label/form-group
      const looksLikeSelect =
        (await nativeSelectLocator.count()) > 0 &&
        (await nativeSelectLocator.evaluate((el: Element) => el.tagName === 'SELECT').catch(() => false));

      if (!looksLikeSelect && dropdownIdentifier) {
        let inferred = page.locator(`select#${dropdownIdentifier}`);
        if ((await inferred.count()) === 0) {
          inferred = page.locator(`//label[@for='${dropdownIdentifier}']/following-sibling::select`);
        }
        if ((await inferred.count()) === 0) {
          inferred = page.locator(`.form-group:has(label:has-text('${dropdownIdentifier}')) select`);
        }
        if ((await inferred.count()) > 0) {
          console.log(`[option_select] Using inferred native select locator for '${displayIdentifier}'.`);
          nativeSelectLocator = inferred;
        }
      }

      if ((await nativeSelectLocator.count()) === 0) {
        console.log(`[option_select] No native HTML <select> element found or matched for '${displayIdentifier}'.`);
      } else {
        await nativeSelectLocator.waitFor({ state: 'visible', timeout: 10000 });
        const isSelectTag = await nativeSelectLocator.evaluate((el: HTMLSelectElement) => el.tagName === 'SELECT').catch(() => false);
        const isEnabled  = await nativeSelectLocator.isEnabled();

        if (isSelectTag && isEnabled) {
          console.log(`[option_select] Confirmed native HTML <select> for '${displayIdentifier}'. Trying strategies...`);

          // Snapshot options for diagnostics and fallback matching
          const optionEls = await nativeSelectLocator.locator('option').all();
          const options = await Promise.all(optionEls.map(async o => ({
            value: await o.getAttribute('value'),
            label: (await o.textContent())?.trim() || ''
          })));
          const labels = options.map(o => o.label);
          const values = options.map(o => o.value);

          // Helper: pick by value equality
          const findByValueEq = (v: string) => options.find(o => (o.value ?? '') === v)?.value ?? null;

          // 0) Try exact VALUE first (critical fix)
          try {
            if (optionText && values.includes(optionText)) {
              await nativeSelectLocator.selectOption({ value: optionText });
              console.log(`[option_select] Selected by exact value='${optionText}'.`);
              return { success: true, message: `Selected '${optionText}' (by value) from native select '${displayIdentifier}'.` };
            }
          } catch (e: any) {
            console.log(`[option_select] Exact value selection failed for '${optionText}': ${e.message}`);
          }

          // 1) Exact LABEL
          try {
            if (optionText && labels.includes(optionText)) {
              await nativeSelectLocator.selectOption({ label: optionText });
              console.log(`[option_select] Selected by exact label='${optionText}'.`);
              return { success: true, message: `Selected '${optionText}' (by label) from native select '${displayIdentifier}'.` };
            }
          } catch (e: any) {
            console.log(`[option_select] Exact label selection failed for '${optionText}': ${e.message}`);
          }

          // 2) Direct scan for VALUE equality (even if Playwright didn't find it)
          const eqVal = optionText ? findByValueEq(optionText) : null;
          if (eqVal) {
            await nativeSelectLocator.selectOption({ value: eqVal });
            console.log(`[option_select] Selected via manual scan by value='${eqVal}'.`);
            return { success: true, message: `Selected '${optionText}' (by value scan) from native select '${displayIdentifier}'.` };
          }

          // 3) Heuristic: last 4/6 digits → match in label (masked accounts)
          const digits = (optionText || '').replace(/\D+/g, '');
          const last6 = digits.slice(-6);
          const last4 = digits.slice(-4);
          const endsWithDigits = (lbl: string, d: string) => d && new RegExp(`${d}\\)?\\s*$`).test(lbl); // handles "...4455)" etc.

          let candidate = options.find(o => endsWithDigits(o.label, last6)) ??
                          options.find(o => endsWithDigits(o.label, last4));
          if (!candidate && digits) {
            // Fallback: contains digit run anywhere
            candidate = options.find(o => o.label.replace(/\D+/g, '').includes(last6) && last6) ??
                        options.find(o => o.label.replace(/\D+/g, '').includes(last4) && last4);
          }
          if (candidate?.value) {
            await nativeSelectLocator.selectOption({ value: candidate.value });
            console.log(`[option_select] Selected via digit heuristic (label '${candidate.label}', value='${candidate.value}').`);
            return { success: true, message: `Selected '${optionText}' (digit heuristic) from native select '${displayIdentifier}'.` };
          }

          // 4) Case-insensitive substring in LABEL
          const lower = (optionText || '').toLowerCase();
          const contains = options.find(o => o.label.toLowerCase().includes(lower));
          if (contains?.value) {
            await nativeSelectLocator.selectOption({ value: contains.value });
            console.log(`[option_select] Selected via label contains match ('${contains.label}').`);
            return { success: true, message: `Selected '${optionText}' (label contains) from native select '${displayIdentifier}'.` };
          }

          // Diagnostics when nothing matched
          console.log(`[option_select] Available option values: ${JSON.stringify(values)}; labels: ${JSON.stringify(labels)}`);
          return { success: false, message: `Option '${optionText}' not found in native select '${displayIdentifier}'.` };
        } else {
          console.log(`[option_select] Native select found but invalid or disabled: isSelectTag=${isSelectTag}, isEnabled=${isEnabled}`);
        }
      }
    } catch (e: any) {
      console.log(`[option_select] Error during native select interaction for '${displayIdentifier}' (client selector '${clientSelector}'): ${e.message}. Proceeding to React-Select strategy.`);
    }

    // --- Strategy 2: Attempt to handle React-Select component ---
    console.log(`[option_select] Strategy 2: Checking for React-Select component for '${displayIdentifier}'...`);
    
    // Initialize reactSelectControlLocator to a dummy locator to ensure it's never null
    let reactSelectControlLocator: Locator = page.locator('__initial_dummy_locator__');

    try {
      // Priority 1: Use the client's provided selector directly, IF it targets a known clickable container/control.
      try {
        const primaryControlCandidate = eval(`page.${clientSelector}`);
        if (await primaryControlCandidate.count() > 0) {
            const isPotentialDirectControl = await primaryControlCandidate.evaluate((el: Element) => {
                const classes = el.classList;
                return classes.contains('Select-control') || classes.contains('Select-multi-value-wrapper') || classes.contains('Select-placeholder');
            }).catch(() => false);

            if (isPotentialDirectControl) {
                reactSelectControlLocator = primaryControlCandidate;
                console.log(`[option_select] Using client-provided selector as primary React-Select control (container): '${clientSelector}'`);
            }
        }
      } catch (e: any) {
        console.warn(`[option_select] Eval failed for client selector as potential React-Select control '${clientSelector}': ${e.message}. Trying fallbacks.`);
      }

      // Priority 2: Fallback to Playwright's built-in locators for combobox or label, which often target the input.
      if (await reactSelectControlLocator.count() === 0) { // Only proceed if no control found yet
        let inputComboboxLocator: Locator = page.locator('__input_combobox_dummy__'); // Initialize dummy

        if (dropdownIdentifier) {
            inputComboboxLocator = page.getByLabel(dropdownIdentifier).or(
                                   page.getByRole('combobox', { name: dropdownIdentifier }));
            if (await inputComboboxLocator.count() === 0) {
                inputComboboxLocator = page.locator(`.form-group:has(label:has-text('${dropdownIdentifier}')) input[role='combobox']`);
            }
        } else {
            inputComboboxLocator = page.getByRole('combobox').first();
        }

        if (await inputComboboxLocator.count() > 0) {
            await inputComboboxLocator.waitFor({ state: 'visible', timeout: 5000 }).catch(() => {});
            const isInputVisible = await inputComboboxLocator.isVisible();
            const isInputEnabled = await inputComboboxLocator.isEnabled();
            
            if (isInputVisible && isInputEnabled) {
                 const parentLocator = inputComboboxLocator.locator('..').filter({has: inputComboboxLocator});
                 if (await parentLocator.count() > 0) {
                     const isParentContainer = await parentLocator.evaluate((el: Element) => {
                         const classes = el.classList;
                         return classes.contains('Select-control') || classes.contains('Select-multi-value-wrapper');
                     }).catch(() => false);
                     
                     if (isParentContainer) {
                         reactSelectControlLocator = parentLocator;
                         console.log(`[option_select] Found input combobox but clicking its parent container.`);
                     } else {
                         reactSelectControlLocator = inputComboboxLocator;
                         console.log(`[option_select] Using input combobox as React-Select control (parent not known container).`);
                     }
                 } else {
                     reactSelectControlLocator = inputComboboxLocator;
                     console.log(`[option_select] Using input combobox as React-Select control (no identifiable parent).`);
                 }
            } else {
                 reactSelectControlLocator = inputComboboxLocator;
                 console.log(`[option_select] Using input combobox as React-Select control (not visible/enabled).`);
            }
        }
      }

      // Final attempt to find a control if all specific attempts failed
      if (await reactSelectControlLocator.count() === 0) { // Only proceed if no control found yet
          if (dropdownIdentifier) {
              reactSelectControlLocator = page.locator(`.Select-control`).filter({hasText: dropdownIdentifier}).first();
              if (await reactSelectControlLocator.count() === 0) {
                  reactSelectControlLocator = page.locator(`.Select-multi-value-wrapper`).filter({hasText: dropdownIdentifier}).first();
              }
          }
          if (await reactSelectControlLocator.count() === 0) {
             reactSelectControlLocator = page.locator(`.Select-control`).or(page.locator(`.Select-multi-value-wrapper`)).first();
          }
      }


      if (await reactSelectControlLocator.count() > 0) { // Now reactSelectControlLocator is guaranteed to be a Locator
          await reactSelectControlLocator.waitFor({ state: 'visible', timeout: 10000 });
          
          if (await reactSelectControlLocator.isVisible() && await reactSelectControlLocator.isEnabled()) {
              console.log(`[option_select] Found final React-Select control for '${displayIdentifier}'. Clicking to open dropdown...`);
              await reactSelectControlLocator.click();

              let optionLocator: Locator = page.locator(`.Select-option:has-text('${optionText}')`);
              if (await optionLocator.count() === 0) {
                  optionLocator = page.getByText(optionText, { exact: true }).first();
              }

              if (await optionLocator.count() > 0) {
                  await optionLocator.waitFor({ state: 'visible', timeout: 5000 });
                  console.log(`[option_select] Found option '${optionText}'. Clicking option...`);
                  await optionLocator.click();
                  console.log(`[option_select] Successfully selected '${optionText}' from React-Select '${displayIdentifier}'.`);
                  return { success: true, message: `Selected '${optionText}' from React-Select '${displayIdentifier}'.` };
              } else {
                  console.log(`[option_select] Error: Could not find option '${optionText}' for React-Select '${displayIdentifier}' after opening the dropdown.`);
                  return { success: false, message: `Option '${optionText}' not found for React-Select '${displayIdentifier}'.` };
              }
          } else {
              console.log(`[option_select] React-Select control '${displayIdentifier}' found but not visible or enabled.`);
              return { success: false, message: `React-Select control '${displayIdentifier}' not visible or enabled.`};
          }
      } else {
          console.log(`[option_select] No suitable React-Select control found for '${displayIdentifier}'.`);
          return { success: false, message: `React-Select control for '${displayIdentifier}' not found.` };
      }
    } catch (e: any) {
      console.log(`[option_select] An unexpected error occurred during React-Select interaction for '${displayIdentifier}': ${e.message}`);
      return { success: false, message: `Failed to select option '${optionText}' from '${displayIdentifier}'. Error: ${e.message}` };
    }

    console.error(`[option_select] Failed to select option '${optionText}' from '${displayIdentifier}' using any strategy. Original client selector: '${clientSelector}'.`);
    return { success: false, message: `Could not find or interact with any dropdown for '${displayIdentifier}' using either native or React-Select strategies.` };
  },
};


// const browserCheckTool: Tool = {
//   name: 'browser_check',
//   description: 'Check a checkbox or radio button',
//   parameters: {
//     type: 'object',
//     properties: {
//       selector: { type: 'string', description: 'Selector for the checkbox/radio' }
//     },
//     required: ['selector']
//   },
//   async execute(ctx, args) {
//     const locator: Locator = await eval(`ctx.page.${args.selector}`);
//     await locator.check();
//     return { success: true, message: `Checked ${args.selector}` };
//   }
// };

// const browserUncheckTool: Tool = {
//   name: 'browser_uncheck',
//   description: 'Uncheck a checkbox',
//   parameters: {
//     type: 'object',
//     properties: {
//       selector: { type: 'string', description: 'Selector for the checkbox' }
//     },
//     required: ['selector']
//   },
//   async execute(ctx, args) {
//     const locator: Locator = await eval(`ctx.page.${args.selector}`);
//     await locator.uncheck();
//     return { success: true, message: `Unchecked ${args.selector}` };
//   }
// };



const browserCheckTool: Tool = {
  name: 'browser_check',
  description: 'Checks one or more checkboxes or radio buttons. Each item in the `selector` array should be a Playwright locator expression (e.g., "getByLabel(\'Remember me\')", "locator(\'.my-checkbox\')"). If a locator expression resolves to multiple elements, all matched elements will be checked.',
  parameters: {
    type: 'object',
    properties: {
      selector: {
        type: 'array', // Changed to array type
        items: {
          type: 'string' // Array of strings, where each string is a Playwright locator expression
        },
        description: 'An array of Playwright locator expressions for the checkboxes/radio buttons to check.'
      }
    },
    required: ['selector']
  },
  async execute(ctx, args) {
    const selectorExpressions: string[] = args.selector;
    const results: { expression: string; success: boolean; message: string }[] = [];

    for (const expr of selectorExpressions) {
      try {
        // Evaluate the expression string to get a Playwright Locator object.
        // WARNING: Using 'eval' with untrusted input can be a security risk.
        // Ensure that the 'expr' strings are controlled or sanitized if coming from external sources.
        const locator: Locator = await eval(`ctx.page.${expr}`);

        // Playwright's locator.check() automatically checks all elements found by the locator
        // if the locator itself resolves to multiple elements.
        await locator.check();
        results.push({ expression: expr, success: true, message: `Successfully checked element(s) for expression: ${expr}` });
      } catch (error: any) {
        results.push({ expression: expr, success: false, message: `Failed to check element(s) for expression "${expr}": ${error.message}` });
      }
    }

    // Determine overall success and create a summary message
    const allSucceeded = results.every(r => r.success);
    const summaryMessage = results.map(r => r.message).join('\n');

    return {
      success: allSucceeded, // Overall success indicates if all operations succeeded
      message: summaryMessage // Detailed message for each expression
    };
  }
};

const browserUncheckTool: Tool = {
  name: 'browser_uncheck',
  description: 'Unchecks one or more checkboxes. Each item in the `selector` array should be a Playwright locator expression (e.g., "getByLabel(\'Remember me\')", "locator(\'.my-checkbox\')"). If a locator expression resolves to multiple elements, all matched elements will be unchecked.',
  parameters: {
    type: 'object',
    properties: {
      selector: {
        type: 'array', // Changed to array type
        items: {
          type: 'string' // Array of strings, where each string is a Playwright locator expression
        },
        description: 'An array of Playwright locator expressions for the checkboxes to uncheck.'
      }
    },
    required: ['selector']
  },
  async execute(ctx, args) {
    const selectorExpressions: string[] = args.selector;
    const results: { expression: string; success: boolean; message: string }[] = [];

    for (const expr of selectorExpressions) {
      try {
        // Evaluate the expression string to get a Playwright Locator object.
        // WARNING: Using 'eval' with untrusted input can be a security risk.
        // Ensure that the 'expr' strings are controlled or sanitized if coming from external sources.
        const locator: Locator = await eval(`ctx.page.${expr}`);

        // Playwright's locator.uncheck() automatically unchecks all elements found by the locator
        // if the locator itself resolves to multiple elements.
        await locator.uncheck();
        results.push({ expression: expr, success: true, message: `Successfully unchecked element(s) for expression: ${expr}` });
      } catch (error: any) {
        results.push({ expression: expr, success: false, message: `Failed to uncheck element(s) for expression "${expr}": ${error.message}` });
      }
    }

    // Determine overall success and create a summary message
    const allSucceeded = results.every(r => r.success);
    const summaryMessage = results.map(r => r.message).join('\n');

    return {
      success: allSucceeded, // Overall success indicates if all operations succeeded
      message: summaryMessage // Detailed message for each expression
    };
  }
};








const browserAssertValueTool: Tool = {
  name: 'browser_assert_value',
  description: 'Assert that an <input>, <textarea>, or <select> element has the expected value.',
  parameters: {
    type: 'object',
    properties: {
      // Changed from 'selector' to 'selectors' and type to 'array'
      selectors: { 
        type: 'array',
        items: { type: 'string' },
        description: 'An array of Playwright locator expressions, ordered by priority. The tool will try them sequentially until one succeeds.'
      },
      expected: { type: 'string', description: 'Expected value' },
    },
    required: ['selectors', 'expected'], // 'selectors' is now required
  },
  async execute(ctx, args) { 
    const { selectors, expected } = args as { selectors: string[]; expected: string }; // Expecting an array
    const timeout = 15000; // Define a standard timeout in milliseconds (using your increased timeout)

    if (!selectors || selectors.length === 0) {
      throw new Error("No selectors provided for browser_assert_value tool.");
    }

    let lastError: any = null; // Variable to store the last error message

    for (const selector of selectors) { // Loop through provided selectors
      let locator: Locator;
      try {
        // --- THIS SECTION IS DESIGNED TO HANDLE ALL POSSIBLE SELECTOR STRING FORMATS ---
        // If the selector string starts with "page.", assume it's a direct Playwright method call.
        // Otherwise, treat it as a simple CSS selector to be passed to page.locator().
        const expr = selector.startsWith('page.')
          ? selector.replace(/^page\./, 'ctx.page.') // Replace 'page.' with 'ctx.page.' for evaluation context
          : `ctx.page.locator('${selector}')`; // Wrap simple CSS selectors with ctx.page.locator()

        locator = await eval(expr); // Evaluate the string to get the actual Playwright Locator object
        // --- END OF SELECTOR HANDLING SECTION ---

        // 1. Wait for the element to be attached to the DOM (for robustness)
        await locator.waitFor({ state: 'attached', timeout: timeout });
        // 2. Wait for the element to be visible/editable (more specific for inputs)
        await locator.waitFor({ state: 'visible', timeout: timeout });

        let actualValue: string | null = null;
        const tagName = (await locator.evaluate(el => el.tagName)).toUpperCase();

        if (tagName === 'INPUT' || tagName === 'TEXTAREA') {
          // For <input> and <textarea> elements, use inputValue()
          actualValue = await locator.inputValue();
        } else if (tagName === 'SELECT') {
          // For <select> elements, get the value of the selected option
          actualValue = await locator.evaluate((el: HTMLSelectElement) => el.value);
        } else {
          // For other elements, default to textContent.
          // While 'assertValue' usually implies inputs, this provides a fallback.
          actualValue = await locator.textContent();
        }

        if (actualValue === expected) {
          // Success! Return immediately if a selector works and value matches
          return { success: true, message: `Value matches "${expected}" using selector "${selector}"`,succeededSelector:selector };
        } else {
          // If value doesn't match, this is a failure for this selector
          throw new Error(
            `Assertion failed: field ${selector} value="${actualValue}", expected="${expected}"`
          );
        }
      } catch (error: any) {
        lastError = error; // Store the error from the current failed selector
        // If this selector failed, try the next one in the loop.
        // Log a warning so you know which selectors were attempted and failed.
        console.warn(`Selector "${selector}" failed for assertValue: ${error.message}. Trying next selector...`);
      }
    }
    // If the loop finishes without returning, it means all selectors failed.
    throw new Error(
      `Tool execution failed: All provided selectors for assertValue failed for expected value "${expected}". ` +
      `Last error: ${lastError?.message || 'Unknown error occurred trying selectors.'}`
    );
  },
};

export const browserAssertSelectedOptionTool: Tool = {
  name: 'browser_assert_selected_option',
  description: 'Assert that a dropdown (native <select> or React-Select) has the expected selected option value/text.',
  parameters: {
    type: 'object',
    properties: {
      selector: { type: 'string', description: 'Playwright selector string for the dropdown or its main input component (e.g., for React-Select)' },
      expected: { type: 'string', description: 'Expected selected option text or value' },
    },
    required: ['selector', 'expected'],
  },
  async execute(ctx, args) {
    const { selector, expected } = args as { selector: string; expected: string };
    try {
      const rawSelector = selector.trim();
      let expr: string;

      if (rawSelector.startsWith('page.')) {
        expr = rawSelector.replace(/^page\./, 'ctx.page.');
      } else if (rawSelector.startsWith('locator(') || rawSelector.startsWith('getBy')) {
        expr = `ctx.page.${rawSelector}`;
      } else {
        expr = `ctx.page.locator('${rawSelector}')`;
      }

      const locator: Locator = await eval(expr);
      await locator.waitFor({ state: 'visible', timeout: 5000 });

      // Try to get value if it's an input or textarea
      const tagName = await locator.evaluate(el => el.tagName.toLowerCase());
      let actualValue: string | null = null;

      if (tagName === 'select') {
        // Native select element
        actualValue = await locator.evaluate((el: HTMLSelectElement) => el.value);
        if (actualValue !== expected) {
            throw new Error(`Assertion failed: Native select ${selector} has value="${actualValue}", expected="${expected}"`);
        }
      } else if (tagName === 'input' || tagName === 'textarea') {
        // Likely a React-Select combobox input or similar.
        // Try getting the input's 'value' attribute
        actualValue = await locator.inputValue(); // Playwright's helper for input value
        if (actualValue !== expected) {
            // If the input value doesn't match, try to find the displayed text
            // React-Select's main input often doesn't hold the full text, but an internal value.
            // The displayed text is usually in a sibling/child span/div.

            // A common pattern for React-Select is that the selected value is displayed in a sibling `div` or `span`
            // of the 'combobox' input, often with a specific class.
            // Or, you might directly assert on the text content of the locator itself if it's a display element.

            // Based on your DOM, the input with #react-select-2--value has role="combobox".
            // The *displayed* text 'Quarterly' is a sibling span:
            // "page.getByText('Quarterly')", "page.getByRole('option', { name: 'Quarterly'})", "page.locator('#react-select-2--value-item')"
            // A more robust check for React-Select would be to check if the `expected` text is visible within the component's visible area.
            // Or, if the LLM provided a selector for the *displayed text* itself, use that.

            // For now, let's try asserting the text content of the locator itself,
            // or a child element if the input itself doesn't contain the visible text.
            const visibleText = await locator.textContent(); // Get text content of the combobox input area.
            if (visibleText && visibleText.includes(expected)) {
                return { success: true, message: `Displayed text for ${selector} contains "${expected}"` };
            } else {
                throw new Error(`Assertion failed: Element ${selector} (tag: ${tagName}) has value="${actualValue}" and visible text "${visibleText}", expected displayed text or value "${expected}"`);
            }
        }
      } else {
        // It's neither a select nor a common input/textarea that holds the value directly.
        // Try to get its text content, which might be the displayed selected option.
        actualValue = await locator.textContent();
        if (actualValue && actualValue.trim() === expected) {
            return { success: true, message: `Text content for ${selector} matches "${expected}"` };
        } else {
            throw new Error(`Assertion failed: Element ${selector} (tag: ${tagName}) has text="${actualValue}", expected="${expected}"`);
        }
      }

      return { success: true, message: `Selected option matches "${expected}"` };
    } catch (e: any) {
        throw new Error(
          `Tool execution failed for selector "${selector}": ${e.message}`
        );
    }
  },
};


export const browserAssertDisplayedOptionTextTool: Tool = {
  name: 'browser_assert_displayed_option_text',
  description: 'Asserts that the text content of an element matches the expected value (e.g., for custom dropdowns).',
  parameters: {
    type: 'object',
    properties: {
      selectors: { type: 'array', items: { type: 'string' }, description: 'Array of Playwright selectors for the element displaying the text' },
      expected: { type: 'string', description: 'Expected displayed text (actual value from last_input_values)' },
    },
    required: ['selectors', 'expected'],
  },
  // … your metadata stays the same …
  async execute(ctx, args) {
    const { selectors, expected } = args as { selectors: string[]; expected: string };
    if (!selectors.length) throw new Error("No selectors provided.");

    let lastError: string | null = null;

    for (let sel of selectors) {
      let locator;
      try {
        sel = sel.trim();
        const fnCall = sel.match(/^(?:page\.)?([a-zA-Z_$][\w$]*)\(([\s\S]*)\)$/);
        if (fnCall) {
          const [_, method, argsText] = fnCall;
          const argsList = eval(`[${argsText}]`);
          locator = (ctx.page as any)[method]?.(...argsList);
          if (!locator) throw new Error(`Page has no method "${method}"`);
        } else {
          locator = ctx.page.locator(sel);
        }

        await locator.waitFor({ state: 'visible', timeout: 5000 });
        const actual = (await locator.textContent())?.trim();
        if (actual === expected) {
          return { success: true, message: `Selector "${sel}" displayed "${actual}".` };
        }
        lastError = `Selector "${sel}" found "${actual}", expected "${expected}".`;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        lastError = `Selector "${sel}" failed: ${msg}`;
      }
    }

    throw new Error(
      `Assertion failed. Expected "${expected}". No selector matched. Last error: ${lastError}`
    );
  },
};


// New Tool: browser_assert_displayed_option_text


const browserAssertTextVisibleTool: Tool = {
  name: 'browser_assert_text_visible',
  description: 'Assert that text is visible, optionally in any of several scopes',
  parameters: {
    type: 'object',
    properties: {
      selectors: {
        type: 'array',
        items: { type: 'string' },
        description: 'Optional array of locator expressions to scope into'
      },
      text: { type: 'string', description: 'Text to assert visible' }
    },
    required: ['text']
  },
  async execute(ctx, args) {
    const { selectors, text } = args as { selectors?: string[]; text: string };
    const timeout = 5000;

    if (selectors && selectors.length) {
      let lastError: any;
      for (const sel of selectors) {
        try {
          const raw = sel.trim();
          const expr = raw.startsWith('page.')
            ? raw.replace(/^page\./, 'ctx.page.')
            : raw.startsWith('locator(') || raw.startsWith('getBy')
              ? `ctx.page.${raw}`
              : `ctx.page.locator('${raw}')`;

          let locator: Locator;

          // --- Special handling for getByText to bypass strict mode initially ---
          if (raw.includes("getByText(") || raw.includes("getByRole(") && raw.includes("name:")) {
            // For getByText or getByRole with a name (which also uses text internally),
            // we will try to find if ANY instance of this text/role is visible.
            // This prevents strict mode violation when multiple elements have the same text.
            const allMatchingLocators = await ctx.page.locator(`text=${text}`).all(); // Find all elements with this text
            let foundVisibleMatch = false;

            for (const matchingLocator of allMatchingLocators) {
                try {
                    await matchingLocator.waitFor({ state: 'visible', timeout: timeout / 3 }); // Give it some time
                    // Also verify the text content matches the exact text, especially for getByText which can be fuzzy
                    const actualContent = await matchingLocator.textContent();
                    const actualAriaLabel = await matchingLocator.getAttribute('aria-label');
                    const actualPlaceholder = await matchingLocator.getAttribute('placeholder');

                    if (actualContent?.includes(text) || actualAriaLabel?.includes(text) || actualPlaceholder?.includes(text)) {
                        foundVisibleMatch = true;
                        break; // Found at least one visible match with the content, good to go
                    }
                } catch (e) {
                    // This specific element isn't visible or doesn't have the text, try next
                }
            }

            if (foundVisibleMatch) {
                return { success: true, message: `Asserted "${text}" visible via getByText/getByRole (found at least one visible instance)` };
            } else {
                throw new Error(`Could not find visible text "${text}" via getByText/getByRole within current selector context.`);
            }

          } else {
            // --- Original logic for other specific locators (e.g., #id, [type="..."]) ---
            // These are expected to resolve to a single element.
            locator = await eval(expr);
            await locator.waitFor({ state: 'visible', timeout: timeout });

            // Try Playwright's hasText or containsText first for more general locators
            try {
                // This checks if the 'locator' (e.g., #id) itself contains the text or has a descendant with the text.
                await locator.locator(`:scope:has-text("${text}")`).waitFor({ state: 'attached', timeout: timeout / 3 });
                return { success: true, message: `Asserted "${text}" visible in ${sel} using has-text` };
            } catch (e) {
                // If has-text fails, fall through to check specific attributes
            }

            const tagName = (await locator.evaluate(el => el.tagName)).toUpperCase();
            let foundContent = '';

            if (tagName === 'INPUT' || tagName === 'TEXTAREA') {
              const elType = await locator.evaluate(el => (el as HTMLInputElement).type || 'text');
              if (elType === 'checkbox' || elType === 'radio') {
                  foundContent = await locator.evaluate(el => (el as HTMLInputElement).value || '');
              } else {
                  foundContent = await locator.evaluate(el => (el as HTMLInputElement).placeholder || '');
                  if (!foundContent) {
                      foundContent = await locator.evaluate(el => el.getAttribute('aria-label') || '');
                  }
              }
            } else if (tagName === 'SELECT') {
                foundContent = await locator.evaluate(el => {
                    const selectEl = el as HTMLSelectElement;
                    return selectEl.options[selectEl.selectedIndex]?.textContent || '';
                });
            } else {
              foundContent = await locator.textContent() || '';
            }

            if (foundContent.includes(text)) {
              return { success: true, message: `Asserted "${text}" visible in ${sel} via attribute check` };
            }
          }

        } catch (e: any) {
          lastError = e;
        }
      }
      throw new Error(
        `Could not find text "${text}" in any of selectors ${JSON.stringify(selectors)}\n` +
        `Last error: ${lastError?.message || lastError}`
      );
    } else {
      // ── Fallback: global getByText ─────────────────────────────────
      // When no specific selectors are provided, assume a global text search.
      // Use locator(`text=${text}`).all() to get all matching elements, then check for visibility.
      const allGlobalLocators = ctx.page.locator(`text=${text}`);
      const visibleElements = await allGlobalLocators.all(); // Get all matching locators
      let foundVisible = false;

      for (const el of visibleElements) {
          try {
              await el.waitFor({ state: 'visible', timeout: timeout / 2 }); // Give some time for each to become visible
              foundVisible = true;
              break; // Found at least one visible instance globally
          } catch (e) {
              // Not visible, continue to next
          }
      }

      if (foundVisible) {
          return { success: true, message: `Asserted "${text}" visible globally (found at least one)` };
      } else {
          throw new Error(`Could not find visible text "${text}" globally.`);
      }
    }
  }
};


// Inside your tools.ts or similar file on the MCP server

export const browserAssertElementVisibleTool: Tool = {
  name: 'browser_assert_element_visible',
  description: 'Assert that an element identified by Playwright selectors is visible on the page, trying multiple selectors.',
  parameters: {
    type: 'object',
    properties: {
      selectors: { // <--- Change 'selector' to 'selectors' and type to 'array'
        type: 'array',
        items: { type: 'string' },
        description: 'An array of Playwright locator expressions for the element to assert visibility for, in priority order.',
      },
      timeout: {
        type: 'number',
        description: 'Optional. Maximum time in milliseconds to wait for the element to become visible (default: 5000).',
        default: 5000,
      }
    },
    required: ['selectors'], // <--- Update required field
  },
  async execute(ctx, args) {
    const { selectors, timeout } = args as { selectors: string[]; timeout?: number }; // <--- Destructure 'selectors' as an array

    if (!selectors || selectors.length === 0) {
      throw new Error('No selectors provided for assertion.');
    }

    let lastError: any = null;
    for (const sel of selectors) { // <--- Loop through selectors
      try {
        const rawSelector = sel.trim(); // Now .trim() works on each string in the array

        let expr: string;
        if (rawSelector.startsWith('page.')) {
          expr = rawSelector.replace(/^page\./, 'ctx.page.');
        } else if (rawSelector.startsWith('locator(') || rawSelector.startsWith('getBy')) {
          expr = `ctx.page.${rawSelector}`;
        } else {
          expr = `ctx.page.locator('${rawSelector}')`;
        }

        const locator = await eval(expr);
        await locator.waitFor({ state: 'visible', timeout: timeout });
        return { success: true, message: `Element "${sel}" is visible.` }; // Return on first success
      } catch (e: any) {
        lastError = e;
        console.warn(`Visibility check failed for selector "${sel}": ${e.message}`);
      }
    }
    // If loop finishes without returning, all selectors failed
    throw new Error(
      `Element visibility assertion failed: All selectors failed: ${JSON.stringify(selectors)}\n` +
      `Last error: ${lastError?.message || lastError}`
    );
  },
};


const browserWaitForSelectorTool: Tool = {
  name: 'browser_wait_for_selector',
  description: 'Wait for any of multiple selectors to appear',
  parameters: {
    type: 'object',
    properties: {
      selectors: {
        type: 'array',
        items: { type: 'string' },
        description: 'Array of locator expressions to try'
      },
      timeout: { type: 'number', description: 'Timeout in ms' }
    },
    required: ['selectors']
  },
  async execute(ctx, args) {
    const { selectors, timeout } = args as { selectors: string[]; timeout?: number };
    const maxTimeout = timeout ?? 30_000; // Use provided timeout or default

    let lastError: any;
    for (const sel of selectors) {
      try {
        let locator: Locator;

        // --- START OF FIX ---
        // Handle 'locator('...)' or 'page.getBy...' from Python client
        if (sel.startsWith('locator(') && sel.endsWith(')')) {
          // Extract the inner selector, e.g., 'body' from "locator('body')"
          const innerSelector = sel.substring(sel.indexOf("'") + 1, sel.lastIndexOf("'"));
          locator = ctx.page.locator(innerSelector);
        } else if (sel.startsWith('page.')) {
          // This path needs careful handling if 'page.' implies a direct Playwright method call
          // such as 'page.getByRole("button", { name: "Submit" })'.
          // eval() is still risky here. A safer approach would be to parse the method name and arguments
          // or have the client send structured data instead of raw Playwright method calls as strings.
          // For now, given your current use of eval, let's make it work for the `getBy...` cases.
          // This assumes `sel` is a complete, evaluable Playwright expression like `page.getByText('Login')`
          // or `page.getByRole('button', { name: 'Submit' })`.
          const exprToEval = sel.replace(/^page\./, 'ctx.page.');
          locator = await eval(exprToEval); // Still uses eval, but with correct `ctx.page` prefix
        } else {
          // Assume it's a simple CSS selector if no special prefix
          locator = ctx.page.locator(sel);
        }
        // --- END OF FIX ---

        const startTime = Date.now(); // Record start time
        await locator.waitFor({ state: 'visible', timeout: maxTimeout }); // Wait for visibility
        const endTime = Date.now(); // Record end time
        const actualWaitTime = endTime - startTime; // Calculate actual wait time

        return {
          success: true,
          message: `Waited for "${sel}" (actual wait: ${actualWaitTime}ms, max timeout: ${maxTimeout}ms)`,
          actualWaitTime: actualWaitTime,
          succeededSelector: sel
        };
      } catch (e: any) {
        lastError = e;
        console.warn(`Selector "${sel}" failed to appear: ${e.message}. Trying next selector...`);
      }
    }
    throw new Error(
      `None of the selectors appeared within ${maxTimeout}ms: ${JSON.stringify(selectors)}\n` +
      `Last error: ${lastError?.message || lastError}`
    );
  }
};


const browserGetPageContentTool: Tool = {
  name: 'browser_get_page_content',
  description: 'Return full HTML',
  parameters: { type: 'object', properties: {} },
  async execute(ctx) {
    return { success: true, content: await ctx.page.content() };
  }
};
// 1) Extend your DomElementInfo to carry an options array:
interface DomElementInfo {
  tagName:    string;
  id?:        string;
  name?:      string;
  placeholder?: string;
  ariaLabel?: string;
  role?:      string;
  text?:      string;
  value?:     string;
  href?:      string;
  type?:      string;
  selectors?: string[];
  selector?:  string;
 

  // ← NEW:
  options?: Array<{
    value: string;
    label: string;
  }>;
}





const browserGetDomInfoTool: Tool = {
  name: 'browser_get_dom_info',
  description: 'Return structured DOM info, suggested selectors, and form options',
  parameters: { type: 'object', properties: {} },
  async execute(ctx) {
    const elements: DomElementInfo[] = await ctx.page.evaluate(() => {
      type Option = { value: string; label: string };
      interface DomElementInfo {
        tagName: string;
        id?: string;
        name?: string;
        placeholder?: string;
        ariaLabel?: string;
        role?: string;
        text?: string;
        value?: string;
        href?: string;
        type?: string;
        selectors?: string[];
        selector?: string;
        options?: Option[];
        className?: string;
      }

      const out: DomElementInfo[] = [];
      const all = document.querySelectorAll<HTMLElement>(
        'input, button, a, select, textarea, [role], [tabindex]:not([tabindex="-1"]),' +
        'table, thead, tbody, tr, th, td, img, ' + 
        'h1, h2, h3, h4, h5, h6, p, span, div, ul, ol, li, form, header, footer, nav, section, article, aside'
      );

      all.forEach((elRaw, i) => {
        const el = elRaw as HTMLElement;
        const tag = el.tagName.toLowerCase();
        const rect = el.getBoundingClientRect();
        if (el.getAttribute('role') !== 'combobox') {
          const visible =
            rect.width > 0 &&
            rect.height > 0 &&
            rect.top < window.innerHeight &&
            rect.left < window.innerWidth;
          if (!visible && !['input','select','textarea', 'table', 'tbody', 'thead', 'tr', 'td', 'th', 'img',
            'div', 'span', 'p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'ul', 'ol', 'li', 'form', 'header', 'footer', 'nav', 'section', 'article', 'aside'].includes(tag)) return;
        }

        const info: DomElementInfo = { tagName: tag, selectors: [] };
        // if (el instanceof HTMLInputElement || el instanceof HTMLTextAreaElement) {
        //   info.value = el.value; // <--- ADD THIS LINE
        // }
        if (el.className && typeof el.className === 'string') {
            info.className = el.className;
            // Optionally, add a class-based selector
            // This builds a selector like '.class1.class2'
            const classSelector = '.' + el.className.trim().split(/\s+/).join('.');
            // Only add if it's not just a generic class like 'active' or 'hidden'
            // You might want to refine this logic based on your needs
            if (classSelector.length > 1 && !['.active', '.hidden', '.selected'].includes(classSelector)) {
                info.selectors!.push(`page.locator('${classSelector}')`);
            }
        }
        
        
        // ── UNIVERSAL LABEL EXTRACTION & FORM-GROUP SCOPED SELECTOR ─────────────────
        if (['input','select','textarea'].includes(tag)) {
          const group = el.closest('.form-group');
          const lblEl = group?.querySelector('label');
          if (lblEl) {
            // extract only the text node (drop required-asterisk spans)
            const tn = Array.from(lblEl.childNodes).find(n => n.nodeType === Node.TEXT_NODE);
            const rawText = (tn?.textContent || lblEl.textContent || '').trim();
            const clean = rawText.replace(/[\W_]+$/, '').trim();
            if (clean) {
              info.text = clean;
              const s = clean.replace(/'/g, "\\'");
              // 1) existing Playwright label locator
              info.selectors!.push(`page.getByLabel('${s}')`);
              // 2) our new high-priority, container-scoped CSS locator:
              //    find the .form-group whose <label> has that text, then its child input/select/textarea
              info.selectors!.unshift(
                `page.locator('.form-group:has(label:has-text("${s}")) ${tag}')`
              );
            }
          }
        }

        // ── COMBOBOX “aria-activedescendant” FIRST ─────────────────────────
        if (el.getAttribute('role') === 'combobox') {
          const actId = el.getAttribute('aria-activedescendant');
          if (actId) {
            info.selectors!.unshift(`page.locator('#${actId}')`);
          }
        }

        // ── 1) Standard <select> ───────────────────────────────────────────
        if (el instanceof HTMLSelectElement) {
          info.options = Array.from(el.options).map(opt => ({
            value: opt.value,
            label: (opt.textContent||'').trim()
          }));
          if (el.id) {
            info.selectors!.push(`page.locator('select#${el.id}')`);
          } else {
            info.selectors!.push(`page.locator('select:nth-child(${i+1})')`);
          }
        }

        // ── 2) Radio/checkbox groups ─────────────────────────────────────
        if (
          el instanceof HTMLInputElement &&
          (el.type==='radio'||el.type==='checkbox') &&
          el.name &&
          !out.some(x=>x.name===el.name&&x.type===el.type&&x.options)
        ) {
          const group = Array.from(
            document.querySelectorAll<HTMLInputElement>(
              `input[name="${el.name}"][type="${el.type}"]`
            )
          );
          info.name = el.name; info.type = el.type;
          info.options = group.map(inp => {
            const l = document.querySelector(`label[for="${inp.id}"]`);
            const lbl = l?.textContent?.trim()||inp.value;
            return { value: inp.value, label: lbl };
          });
          info.selectors!.push(`page.locator('input[name="${el.name}"]')`);
        }

        // ── 3) Buttons & typed inputs ─────────────────────────────────────
        if (
          (el instanceof HTMLButtonElement||el instanceof HTMLInputElement) &&
          el.getAttribute('type')
        ) {
          info.type = el.getAttribute('type')!;
          const t = info.type.replace(/'/g,"\\'");
          info.selectors!.push(`page.locator('${tag}[type="${t}"]')`);
        }

        // ── 4) Text content ───────────────────────────────────────────────
        const txt = (el.innerText||el.textContent||'').trim();
        if (txt && txt.length<=150) {
          if (!info.text) info.text = txt;
          const s = txt.replace(/'/g,"\\'").replace(/\n+/g," ");
          info.selectors!.push(`page.getByText('${s}')`);
        }

        // ── 5) Role-based ─────────────────────────────────────────────────
        if (el.getAttribute('role')) {
          info.role = el.getAttribute('role')!;
          const n = (info.text||'').replace(/'/g,"\\'");
          info.selectors!.push(`page.getByRole('${info.role}'${n?`, { name: '${n}'}`:''})`);
        }

        // ── 6) Anchor href ────────────────────────────────────────────────
        if (el instanceof HTMLAnchorElement&&el.getAttribute('href')) {
          const h = el.getAttribute('href')!.replace(/'/g,"\\'");
          info.selectors!.unshift(`page.locator('a[href="${h}"]')`);
        }

        // ── 7) ID ─────────────────────────────────────────────────────────
        if (el.id) {
          info.id = el.id;
          info.selectors!.push(`page.locator('#${el.id}')`);
        }

        // ── 8) placeholder ────────────────────────────────────────────────
        if (el.getAttribute('placeholder')) {
          const ph = el.getAttribute('placeholder')!.replace(/'/g,"\\'");
          info.placeholder = el.getAttribute('placeholder')!;
          info.selectors!.push(`page.getByPlaceholder('${ph}')`);
        }

        // ── 9) aria-label ─────────────────────────────────────────────────
        if (el.getAttribute('aria-label')) {
          const al = el.getAttribute('aria-label')!.replace(/'/g,"\\'");
          info.ariaLabel = el.getAttribute('aria-label')!;
          info.selectors!.push(`page.getByLabel('${al}')`);
        }

        // ── fallback nth-child ─────────────────────────────────────────────
        info.selectors!.push(`page.locator('${tag}:nth-child(${i+1})')`);

        info.selector = info.selectors![0];
        out.push(info);
      });

      return out;
    });

    return { success: true, elements };
  }
};



const browserAssertUrlContainsTool: Tool = {
  name: 'browser_assert_url_contains',
  description: 'Assert that the current URL contains a given substring',
  parameters: {
    type: 'object',
    properties: {
      fragment: { type: 'string', description: 'Substring expected in the URL' }
    },
    required: ['fragment']
  },
  async execute(ctx, args) {
    const { fragment } = args as { fragment: string };
    const current = ctx.page.url();
    if (!current.toLowerCase().includes(fragment.toLowerCase())) {
      throw new Error(`URL "${current}" does not contain "${fragment}"`);
    }
    return { success: true, message: `URL contains "${fragment}"` };
  }
};



const browserAssertCellValueInRow: Tool = {
  name: 'browser_assert_cell_value_in_row',
  description: 'Asserts a specific value in a cell of a table row, identified by content in another cell. Can try multiple table selectors.',
  parameters: {
    type: 'object',
    properties: {
      table_selector: {
        type: 'array', // Changed to array
        items: { type: 'string' }, // Items are strings
        description: 'Array of selectors for the table element (e.g., ["table#mainTable", "table.data-grid", "table", "page.locator(\'table#another\')"])'
      },
      row_identifier_column: {
        type: 'string',
        description: 'Text of the column header (or 1-based index) used to uniquely identify the row (e.g., "Applicant Name", 2)'
      },
      row_identifier_value: {
        type: 'string',
        description: 'The value in the row_identifier_column that identifies the specific row (e.g., "Robert Johnson")'
      },
      target_column: {
        type: 'string',
        description: 'Text of the column header (or 1-based index) where the value should be asserted (e.g., "Status", 5)'
      },
      expected_value: {
        type: 'string',
        description: 'The expected text value in the target cell.'
      }
    },
    required: ['table_selector', 'row_identifier_column', 'row_identifier_value', 'target_column', 'expected_value']
  },
  async execute(ctx: any, args: any) {
    const { table_selector, row_identifier_column, row_identifier_value, target_column, expected_value } = args;
    const timeout = 5000;
    let table;
    let foundTable = false;
    let lastTableError: any;

    // Helper function to normalize selector strings
    const normalizeSelector = (sel: string): string => {
        // Remove 'page.' prefix if present
        if (sel.startsWith('page.')) {
            sel = sel.substring(5);
        }
        // Extract content inside locator() if present
        const locatorMatch = sel.match(/^locator\(['"](.*)['"]\)$/);
        if (locatorMatch && locatorMatch[1]) {
            return locatorMatch[1];
        }
        // Return as is if no 'page.' or 'locator()' wrapper
        return sel;
    };

    // Try each table selector until one is found and visible
    for (const rawSel of table_selector) { // Iterate over raw selectors
      const sel = normalizeSelector(rawSel); // Normalize each selector
      try {
        table = ctx.page.locator(sel); // Now 'sel' should be a pure CSS/text/role selector
        // Use a portion of the total timeout for each selector attempt
        await table.waitFor({ state: 'visible', timeout: timeout / table_selector.length });
        foundTable = true;
        break; // Found a visible table, exit loop
      } catch (e: any) {
        lastTableError = e; // Store the last error if no table is found
        // Continue to the next selector
      }
    }

    if (!foundTable || !table) {
      throw new Error(
        `Could not find any visible table using selectors: ${JSON.stringify(table_selector)}.\n` +
        `Last attempt error: ${lastTableError?.message || lastTableError}`
      );
    }

    try {
      // Find the header index for row_identifier_column and target_column
      // This assumes the table has a <thead> with <th> or <td> elements for headers
      const headers = await table.locator('thead th, thead td').allTextContents();
      let identifierColIndex = -1;
      let targetColIndex = -1;

      if (typeof row_identifier_column === 'string') {
        // Find by header text (case-insensitive, trimmed)
        identifierColIndex = headers.findIndex((h: string) => h.trim().toLowerCase() === row_identifier_column.trim().toLowerCase());
      } else if (typeof row_identifier_column === 'number') { // Ensure it's a number
        // Use 0-based index if a number is provided (tool expects 1-based, so convert)
        identifierColIndex = row_identifier_column - 1;
      }

      if (typeof target_column === 'string') {
        // Find by header text (case-insensitive, trimmed)
        targetColIndex = headers.findIndex((h: string) => h.trim().toLowerCase() === target_column.trim().toLowerCase());
      } else if (typeof target_column === 'number') { // Ensure it's a number
        // Use 0-based index if a number is provided (tool expects 1-based, so convert)
        targetColIndex = target_column - 1;
      }

      // Error handling if columns are not found or invalid type
      if (identifierColIndex === -1) {
        throw new Error(`Row identifier column "${row_identifier_column}" not found in table headers or invalid type/index.`);
      }
      if (targetColIndex === -1) {
        throw new Error(`Target column "${target_column}" not found in table headers or invalid type/index.`);
      }

      // Find the row that contains the identifier value in the correct column
      // This locator strategy finds a tr within tbody that has a td at the calculated 1-based index
      // containing the row_identifier_value.
      // Make sure row_identifier_value is properly escaped for the Playwright selector if it contains special characters
      const escapedRowIdentifierValue = row_identifier_value.replace(/'/g, "\\'"); // Basic escaping for single quotes
      const targetRow = table.locator(`tbody tr:has(td:nth-child(${identifierColIndex + 1}):has-text("${escapedRowIdentifierValue}"))`);


      // Wait for the identified row to be visible
      await targetRow.waitFor({ state: 'visible', timeout });

      // Find the target cell within that specific row
      const targetCell = targetRow.locator(`td:nth-child(${targetColIndex + 1})`);

      // Wait for the target cell to be visible
      await targetCell.waitFor({ state: 'visible', timeout });

      // Get the text content of the target cell
      const actualValue = await targetCell.textContent();

      // Assert that the actual value includes the expected value (trimmed)
      if (actualValue?.trim().toLowerCase().includes(expected_value.trim().toLowerCase())){
        return { success: true, message: `Successfully asserted "${expected_value}" in column "${target_column}" for row identified by "${row_identifier_value}".` };
      } else {
        throw new Error(`Expected "${expected_value}" in column "${target_column}" for row identified by "${row_identifier_value}", but found "${actualValue}".`);
      }

    } catch (error: any) {
      // Catch and re-throw any errors with a more descriptive message
      throw new Error(`Failed to assert cell value: ${error.message}`);
    }
  }
};





const browserAssertTableColumnValues: Tool = {
  name: 'browser_assert_table_column_values',
  description: 'Asserts visible values in a specified table column. Can verify if all cells match, if no cells match, or if at least one cell matches an expected value or pattern. Can try multiple table selectors.',
  parameters: {
    type: 'object',
    properties: {
      table_selector: {
        type: 'array',
        items: { type: 'string' },
        description: 'Array of selectors for the table element (e.g., ["table#mainTable", "page.locator(\'table.data-grid\')"])'
      },
      column_header: {
        type: 'string',
        description: 'Text of the column header whose values should be asserted (e.g., "Status"). Prefer exact header text.'
      },
      expected_value: {
        type: 'string',
        description: 'The text value to assert against. Case-insensitive comparison is performed.'
      },
      match_type: {
        type: 'string',
        enum: ['exact', 'includes'],
        default: 'includes',
        description: 'How the expected_value should be matched. "exact" requires the cell content to be exactly the expected value. "includes" requires the cell content to merely contain the expected value. Defaults to "includes".'
      },
      // --- NEW PARAMETER ---
      assertion_type: {
        type: 'string',
        enum: ['all', 'none', 'any'],
        default: 'all', // Default behavior is to assert 'all' cells match
        description: 'Specifies the assertion type: "all" (all cells must match expected_value), "none" (no cells must match expected_value), or "any" (at least one cell must match expected_value).'
      }
      // 'negative_assertion' is now effectively replaced by 'assertion_type: "none"'
    },
    required: ['table_selector', 'column_header', 'expected_value']
  },
  async execute(ctx: any, args: any) {
    const { table_selector, column_header, expected_value, match_type = 'includes', assertion_type = 'all' } = args;
    const timeout = 5000;
    let tableLocator;
    let foundTable = false;
    let lastTableError: any;

    // (normalizeSelector function remains the same as your existing code)
    const normalizeSelector = (sel: string): string => {
        if (sel.startsWith('page.')) { sel = sel.substring(5); }
        const locatorMatch = sel.match(/^locator\(['"](.*)['"]\)$/);
        if (locatorMatch && locatorMatch[1]) { return locatorMatch[1]; }
        return sel;
    };

    for (const rawSel of table_selector) {
      const sel = normalizeSelector(rawSel);
      try {
        tableLocator = ctx.page.locator(sel);
        await tableLocator.waitFor({ state: 'visible', timeout: timeout / table_selector.length });
        foundTable = true;
        break;
      } catch (e: any) {
        lastTableError = e;
      }
    }

    if (!foundTable || !tableLocator) {
      throw new Error(`Could not find any visible table using selectors: ${JSON.stringify(table_selector)}.\nLast attempt error: ${lastTableError?.message || lastTableError}`);
    }

    try {
      const headers = await tableLocator.locator('thead th, thead td').allTextContents();
      let colIndex = -1;

      if (typeof column_header === 'string') {
        colIndex = headers.findIndex((h: string) => h.trim().toLowerCase() === column_header.trim().toLowerCase());
      } else if (typeof column_header === 'number' && column_header >= 1 && column_header <= headers.length) {
        colIndex = column_header - 1;
      }

      if (colIndex === -1) {
        throw new Error(`Column header "${column_header}" not found in table headers or invalid type/index.`);
      }

      const columnCells = await tableLocator.locator(`tbody tr td:nth-child(${colIndex + 1})`).all();

      if (columnCells.length === 0) {
        // Handle cases where no rows are found
        if (assertion_type === 'none') {
          return { success: true, message: `No visible rows found in column "${column_header}". Assertion passed as nothing to not find.` };
        } else { // 'all' or 'any' type assertions where rows are expected
          throw new Error(`No visible rows found in column "${column_header}". Cannot perform "${assertion_type}" assertion for "${expected_value}".`);
        }
      }

      const lowerExpected = expected_value.trim().toLowerCase();
      const violatingCells: string[] = []; // For 'all' and 'none'
      let foundAnyMatch = false; // For 'any'

      for (const cell of columnCells) {
        await cell.waitFor({ state: 'visible', timeout: timeout / columnCells.length });
        const actualValue = (await cell.textContent())?.trim() || '';
        const lowerActual = actualValue.toLowerCase();

        let currentMatch = false;
        if (match_type === 'exact') {
          currentMatch = lowerActual === lowerExpected;
        } else { // 'includes'
          currentMatch = lowerActual.includes(lowerExpected);
        }

        if (assertion_type === 'all') {
          if (!currentMatch) {
            violatingCells.push(`Cell value "${actualValue}" did not match expected "${expected_value}" (match type: ${match_type})`);
          }
        } else if (assertion_type === 'none') {
          if (currentMatch) {
            violatingCells.push(`Cell value "${actualValue}" was found, but should NOT match expected "${expected_value}" (match type: ${match_type})`);
          }
        } else if (assertion_type === 'any') {
          if (currentMatch) {
            foundAnyMatch = true;
            break; // Found one, so no need to check further for 'any'
          }
        }
      }

      // Final assertion logic based on assertion_type
      if (assertion_type === 'all' || assertion_type === 'none') {
        if (violatingCells.length > 0) {
          throw new Error(`Failed to assert all column values in "${column_header}". The following issues were found: ${violatingCells.join('; ')}`);
        } else {
          return { success: true, message: `Successfully asserted that ${assertion_type} visible cells in column "${column_header}" match "${expected_value}" with match type "${match_type}".` };
        }
      } else if (assertion_type === 'any') {
        if (foundAnyMatch) {
          return { success: true, message: `Successfully found at least one cell matching "${expected_value}" in column "${column_header}" with match type "${match_type}".` };
        } else {
          throw new Error(`Expected to find at least one cell matching "${expected_value}" in column "${column_header}" (match type: ${match_type}), but none were found.`);
        }
      }
    } catch (error: any) {
      throw new Error(`Failed to assert table column values: ${error.message}`);
    }
  }
};

const browserAssertFilteredTableRows: Tool = {
  name: 'browser_assert_filtered_table_rows',
  description: 'Asserts values in a specific column for rows that meet certain filter conditions across other columns. Can try multiple table selectors.',
  parameters: {
    type: 'object',
    properties: {
      table_selector: {
        type: 'array',
        items: { type: 'string' },
        description: 'Array of selectors for the table element (e.g., ["table#mainTable", "table.data-grid"])'
      },
      filter_conditions: {
        type: 'array',
        items: {
          type: 'object',
          properties: {
            column: { type: 'string', description: 'Header text or 1-based index of the column to filter by' },
            value: { type: 'string', description: 'Expected value/pattern in the filter column' },
            match_type: { type: 'string', enum: ['exact', 'includes'], default: 'includes', description: 'How to match the filter value (exact or includes)' }
          },
          required: ['column', 'value']
        },
        description: 'Array of conditions to filter rows (e.g., [{ "column": "Status", "value": "Approved" }])'
      },
      assert_column: {
        type: 'string',
        description: 'Header text or 1-based index of the column to assert values in (e.g., "Name")'
      },
      assert_value: {
        type: 'string',
        description: 'The value to assert against in the assert_column for the filtered rows.'
      },
      assert_match_type: {
        type: 'string',
        enum: ['exact', 'includes'],
        default: 'includes',
        description: 'How to match the asserted value (exact or includes)'
      },
      assert_negative_assertion: {
        type: 'boolean',
        description: 'Set to true if verifying the absence of assert_value in the assert_column for filtered rows.',
        default: false
      }
    },
    required: ['table_selector', 'filter_conditions', 'assert_column', 'assert_value', 'assert_negative_assertion']
  },
  async execute(ctx: any, args: any) {
    const { table_selector, filter_conditions, assert_column, assert_value, assert_match_type, assert_negative_assertion } = args;
    const timeout = 5000; // Use a reasonable default timeout
    let table;
    let foundTable = false;
    let lastTableError: any;
    const normalizeSelector = (sel: string): string => {
    if (sel.startsWith('page.')) {
        sel = sel.substring(5);
    }
    const locatorMatch = sel.match(/^locator\(['"](.*)['"]\)$/);
    if (locatorMatch && locatorMatch[1]) {
        return locatorMatch[1];
    }
    return sel;
};

    // 1. Find the table using robust selectors
    for (const rawSel of table_selector) {
      const sel = normalizeSelector(rawSel);
      try {
        table = ctx.page.locator(sel);
        await table.waitFor({ state: 'visible', timeout: timeout / table_selector.length });
        foundTable = true;
        break;
      } catch (e: any) {
        lastTableError = e;
      }
    }

    if (!foundTable || !table) {
      throw new Error(
        `Could not find any visible table using selectors: ${JSON.stringify(table_selector)}.\n` +
        `Last attempt error: ${lastTableError?.message || lastTableError}`
      );
    }

    try {
      const headers = await table.locator('thead th, thead td').allTextContents();
      const getColumnIndex = (col: string | number): number => {
        if (typeof col === 'string') {
          const index = headers.findIndex((h: string) => h.trim().toLowerCase() === col.trim().toLowerCase());
          if (index === -1) throw new Error(`Column header "${col}" not found.`);
          return index;
        } else if (typeof col === 'number' && col >= 1 && col <= headers.length) {
          return col - 1; // Convert 1-based to 0-based index
        }
        throw new Error(`Invalid column identifier: "${col}". Must be string header or 1-based index.`);
      };

      const filterColumnIndices: { colIndex: number; value: string; matchType: 'exact' | 'includes' }[] = [];
      for (const filter of filter_conditions) {
        filterColumnIndices.push({
          colIndex: getColumnIndex(filter.column),
          value: filter.value,
          matchType: filter.match_type || 'includes'
        });
      }

      const assertColIndex = getColumnIndex(assert_column);

      const rows = await table.locator('tbody tr').all();
      let foundMatchingRows = false;
      let assertionFailures: string[] = [];

      for (const row of rows) {
        let rowMatchesAllFilters = true;
        for (const filter of filterColumnIndices) {
          const cell = row.locator(`td:nth-child(${filter.colIndex + 1})`);
          const cellText = (await cell.textContent())?.trim().toLowerCase() || '';
          const filterValueLower = filter.value.trim().toLowerCase();

          if (filter.matchType === 'exact') {
            if (cellText !== filterValueLower) {
              rowMatchesAllFilters = false;
              break;
            }
          } else { // includes
            if (!cellText.includes(filterValueLower)) {
              rowMatchesAllFilters = false;
              break;
            }
          }
        }

        if (rowMatchesAllFilters) {
          foundMatchingRows = true;
          const assertCell = row.locator(`td:nth-child(${assertColIndex + 1})`);
          const assertCellText = (await assertCell.textContent())?.trim().toLowerCase() || '';
          const assertValueLower = assert_value.trim().toLowerCase();

          let assertionResult = false;
          if (assert_match_type === 'exact') {
            assertionResult = (assertCellText === assertValueLower);
          } else { // includes
            assertionResult = assertCellText.includes(assertValueLower);
          }

          if (assert_negative_assertion) {
            if (assertionResult) { // Found something that should NOT match
              assertionFailures.push(
                `Row identified (filtered by: ${JSON.stringify(filter_conditions)}) has ` +
                `"${assert_column}" value "${assertCellText}" which matches "${assert_value}" (match type: ${assert_match_type}), ` +
                `but it should NOT.`
              );
            }
          } else {
            if (!assertionResult) { // Did NOT find something that SHOULD match
              assertionFailures.push(
                `Row identified (filtered by: ${JSON.stringify(filter_conditions)}) has ` +
                `"${assert_column}" value "${assertCellText}" which does NOT match "${assert_value}" (match type: ${assert_match_type}), ` +
                `but it should.`
              );
            }
          }
        }
      }

      if (!foundMatchingRows) {
          // If no rows matched the filter conditions, and it's a negative assertion (meaning we expect no such rows to cause a match for assert_value)
          // this is technically a pass from the perspective of the assert_value check.
          // However, if the intent was "are there *any* rows that meet the filter condition for us to check",
          // then this might be a test setup issue. For now, assume if no rows match filters, it implies
          // no violations of the assert condition on those filtered rows.
          if (assert_negative_assertion) {
            return { success: true, message: `No rows found matching filter conditions. Implicitly passed negative assertion.` };
          } else {
             // If we expect to find rows matching the filter AND then assert something on them, but found no such rows.
             throw new Error(`No rows found matching filter conditions: ${JSON.stringify(filter_conditions)}. Cannot perform positive assertion.`);
          }
      }


      if (assertionFailures.length > 0) {
        throw new Error(`Failed to assert filtered table rows: ${assertionFailures.join('; ')}`);
      }

      return { success: true, message: `Successfully asserted filtered table rows based on conditions: ${JSON.stringify(filter_conditions)}` };

    } catch (error: any) {
      throw new Error(`Failed to assert filtered table rows: ${error.message}`);
    }
  }
};

// --- Tool Registry ---
const tools: Record<string, Tool> = {
  browser_navigate:            browserNavigateTool,
  browser_type:                browserTypeTool,
  browser_click:               browserClickTool,
  browser_select_option:       optionSelectTool,
  browser_check:               browserCheckTool,
  browser_uncheck:             browserUncheckTool,
  browser_assert_text_visible: browserAssertTextVisibleTool,
  browser_wait_for_selector:   browserWaitForSelectorTool,
  browser_get_page_content:    browserGetPageContentTool,
  browser_get_dom_info:        browserGetDomInfoTool,
  browser_assert_value: browserAssertValueTool,
  browser_assert_selected_option: browserAssertSelectedOptionTool,// native option 
  browser_assert_url_contains: browserAssertUrlContainsTool,
  browser_assert_element_visible: browserAssertElementVisibleTool,
  browser_assert_displayed_option_text: browserAssertDisplayedOptionTextTool,
  browser_assert_cell_value_in_row: browserAssertCellValueInRow,
  browser_assert_table_column_values: browserAssertTableColumnValues,
  browser_assert_filtered_table_rows : browserAssertFilteredTableRows
};


// Define the PoolOptions interface, as it's used in the dbConfig
interface PoolOptions {
    host: string;
    user: string;
    password?: string;
    database: string;
    waitForConnections?: boolean;
    connectionLimit?: number;
    queueLimit?: number;
    enableKeepAlive?: boolean;
    keepAliveInitialDelay?: number;
    multipleStatements?: boolean;
    port?: number;
    connectTimeout?: number;
}

// Load environment variables from the .env file in the parent directory
// Ensure the path is correct relative to where this script will be run
dotenv.config({ path: path.resolve(__dirname, '../../.env') });

// Check environment
const isProduction: boolean = process.env.NODE_ENV === 'production';
console.log("Is in AWS (Production/Test):", isProduction);

let initialDbConfig: PoolOptions;
let initialDbPool: mysql.Pool;

// Setup DB config for the initial connection to fetch environment details
// This initial pool is used to get the dynamic environment details
//const initialDbConfig: PoolOptions = isProduction
//    ? {
//        host: process.env.RDS_HOST!,
//        user: process.env.RDS_USER!,
//        password: process.env.RDS_PASS!,
//        database: process.env.RDS_DB!,
//        waitForConnections: true,
//        connectionLimit: 15,
//        queueLimit: 45,
//        enableKeepAlive: true,
//        keepAliveInitialDelay: 0,
//        multipleStatements: true,
//    }
//    : {
//        host: 'localhost',
//        port: 3306,
//        user: process.env.LOCAL_USER!,
//        password: process.env.LOCAL_PASS!,
//        database: process.env.LOCAL_DB!,
//        waitForConnections: true,
//        connectTimeout: 10000,
//        enableKeepAlive: true,
//        multipleStatements: true,
//    };
//
//// Create the initial database pool
//const initialDbPool = mysql.createPool(initialDbConfig);

// Interface for the structure of environment details fetched from the database
interface EnvironmentDetails {
    environmentId: number;
    db_host: string;
    port: string; // Port is often stored as string in DB, convert to number later
    db_user: string;
    db_password: string;
    db_name: string;
}

/**
 * Fetches environment details from the database by ID.
 * @param id The ID of the environment to fetch.
 * @returns A Promise that resolves to EnvironmentDetails or null if not found.
 */
async function getEnvironmentDetailById(id: number): Promise<EnvironmentDetails | null> {
    try {
        // Execute the query to select environment details
        const [rows] = await initialDbPool.execute<mysql.RowDataPacket[]>('SELECT * FROM environmentDetails WHERE environmentId = ? LIMIT 1', [id]);
        // Check if rows exist and return the first one as EnvironmentDetails
        return Array.isArray(rows) && rows.length > 0 ? rows[0] as EnvironmentDetails : null;
    } catch (error) {
        console.error('Error querying environmentDetails:', error);
        throw error; // Re-throw the error for the calling function to handle
    }
}

// Declare DB_CONFIG outside the IIFE so it can be accessed globally or exported
// It will be assigned its value once the environment details are fetched.
// Use a definite assignment assertion (!) to tell TypeScript it will be initialized.
let DB_CONFIG: PoolOptions | null = null;


function isDatabaseConfigured(): boolean {
    return DB_CONFIG !== null;
}

function getDatabaseConfig(): PoolOptions {
    if (!DB_CONFIG) {
        throw new Error('Database configuration not initialized. Agent must connect and provide environment ID first.');
    }
    return DB_CONFIG;
}

// This Promise will resolve when DB_CONFIG has been successfully initialized.
//const dbConfigInitializationPromise = (async () => {
//    try {
//        // Fetch environment details using the initial database connection
//        const environmentDetails = await getEnvironmentDetailById(0);
//        console.log('Environment details fetched from DB:', environmentDetails);
//
//        // Check if environment details were found
//        if (!environmentDetails) {
//            throw new Error('No environment details found for the given ID.');
//        }
//
//        // Assign the constructed DB_CONFIG using the fetched details
//        DB_CONFIG = {
//            host: environmentDetails.db_host,
//            port: parseInt(environmentDetails.port), // Parse port to integer
//            user: environmentDetails.db_user,
//            password: environmentDetails.db_password,
//            database: environmentDetails.db_name
//        };
//
//        console.log("Final DB_CONFIG has been set:", DB_CONFIG);
//
//    } catch (err) {
//        console.error('Error in DB_CONFIG initialization flow:', err);
//        // Re-throw the error so the promise rejects, allowing external code to catch it
//        throw err;
//    }
//})();



// Your utility functions
function makeJsonSerializable(obj: any): any {
  if (obj instanceof Date) {
    return obj.toISOString();
  } else if (typeof obj === 'bigint') {
    return Number(obj);
  } else if (Array.isArray(obj)) {
    return obj.map(makeJsonSerializable);
  } else if (typeof obj === 'object' && obj !== null) {
    const result: any = {};
    for (const key in obj) {
      result[key] = makeJsonSerializable(obj[key]);
    }
    return result;
  }
  return obj;
}

async function getDbSchema(): Promise<Record<string, any>> {
  const schema: Record<string, any> = {};

  if (!isDatabaseConfigured()) {
    throw new Error('Database not configured. Agent must connect with environment ID first.');
  }

  try {
    const conn = await mysql.createConnection(getDatabaseConfig());
    const [tables] = await conn.execute<any[]>('SHOW TABLES');
    const tableKey = Object.keys(tables[0])[0];

    for (const row of tables) {
      const tableName = row[tableKey];
      const [columns] = await conn.execute<any[]>(`DESCRIBE \`${tableName}\``);
      schema[tableName] = columns.map(col => ({
        Field: col.Field,
        Type: col.Type,
        Null: col.Null,
        Key: col.Key,
        Default: col.Default,
        Extra: col.Extra
      }));
    }
    await conn.end();
    console.debug(`[DEBUG] Schema loaded: ${Object.keys(schema).join(', ')}`);
  } catch (e) {
    console.error('[DBSchema Error]', e);
  }
  return schema;
}


async function executeSql(sql: string): Promise<any[]> {
  try {
    console.debug('[DEBUG] Executing SQL:', sql);
    const conn = await mysql.createConnection(getDatabaseConfig());
    const [rows] = await conn.execute<any[]>(sql);
    await conn.end();
    const serializable = makeJsonSerializable(rows);
    console.debug(`[DEBUG] SQL Result count: ${serializable.length} rows`);
    if (serializable.length > 0) {
      console.debug(`[DEBUG] First row sample: ${Object.keys(serializable[0])}`);
    }
    return serializable;
  } catch (e) {
    console.error('[execute_query Error]', e);
    return [];
  }
}

// Your WebSocket message handler
//async function handleWebSocketMessage(ws: any, msg: any) {
//  const method = msg.method;
//  const params = msg.params || {};
//  const response: any = { jsonrpc: '2.0', id: msg.id };
//
//  try {
//    if (method === 'notifications/initialized') return;
//    else if (method === 'DBSchema') {
//      const environmentId = params.environmentId || 1; // Default to 1 if not provided
//      console.log(`DEBUG: Using environment ID: ${environmentId} for DBSchema`);
//      
//      await reinitializeDbConfig(environmentId);
//
//      response.result = await getDbSchema();
//    } else if (method === 'execute_query') {
//      const sql = params.sql || '';
//      response.result = sql.trim() ? await executeSql(sql) : [];
//    } else if (method === 'echo') {
//      response.result = { echoed: params.message || '' };
//    } else {
//      response.error = { code: -32601, message: `Method \"${method}\" not found` };
//    }
//  } catch (err: any) {
//    console.error(`[ERROR] Exception in ${method}:`, err);
//    response.error = {
//      code: -32000,
//      message: err.message,
//      data: err.stack
//    };
//  }
//
//  ws.send(JSON.stringify(response));
//}

async function handleWebSocketMessage(ws: any, msg: any) {
  const method = msg.method;
  const params = msg.params || {};
  const response: any = { jsonrpc: '2.0', id: msg.id };

  try {
    if (method === 'notifications/initialized') return;
    else if (method === 'DBSchema') {
      const environmentId = params.environmentId;
      
      // Enhanced logging for environment ID
      console.log(`DEBUG: DBSchema request received`);
      console.log(`DEBUG: Raw params received:`, JSON.stringify(params, null, 2));
      
      if (environmentId === undefined || environmentId === null) {
        console.error(`CRITICAL ERROR: No environment ID provided in DBSchema request!`);
        console.error(`ERROR: Params received:`, params);
        throw new Error('Environment ID is required but was not provided');
      }
      
      if (typeof environmentId !== 'number' || environmentId <= 0) {
        console.error(`CRITICAL ERROR: Invalid environment ID: ${environmentId} (type: ${typeof environmentId})`);
        throw new Error(`Invalid environment ID: ${environmentId}. Must be a positive number.`);
      }
      
      console.log(`SUCCESS: EXTRACTED ENVIRONMENT ID FROM AGENT: ${environmentId}`);
      console.log(`DEBUG: Environment ID received from agent testcase: ${environmentId}`);
      console.log(`DEBUG: Reinitializing database configuration for environment: ${environmentId}`);
      
      await reinitializeDbConfig(environmentId);
      
      console.log(`DEBUG: Fetching database schema for environment: ${environmentId}`);
      const schema = await getDbSchema();
      console.log(`SUCCESS: Schema fetched successfully for environment ${environmentId}. Tables: ${Object.keys(schema).length}`);

      response.result = schema;
    } else if (method === 'execute_query') {
      const sql = params.sql || '';
      response.result = sql.trim() ? await executeSql(sql) : [];
    } else if (method === 'echo') {
      response.result = { echoed: params.message || '' };
    } else {
      response.error = { code: -32601, message: `Method \"${method}\" not found` };
    }
  } catch (err: any) {
    console.error(`[ERROR] Exception in ${method}:`, err);
    response.error = {
      code: -32000,
      message: err.message,
      data: err.stack
    };
  }

  ws.send(JSON.stringify(response));
}
async function reinitializeDbConfig(environmentId: number) {
  try {
    console.log(`DEBUG: REINITIALIZING DB_CONFIG FOR ENVIRONMENT ID: ${environmentId}`);
    console.log(`DEBUG: Environment ID received from agent: ${environmentId} (type: ${typeof environmentId})`);
    
    // Fetch environment details using the dynamic ID
    console.log(`DEBUG: Querying environmentDetails table for ID: ${environmentId}`);
    const environmentDetails = await getEnvironmentDetailById(environmentId);
    
    if (!environmentDetails) {
      console.error(`ERROR: NO ENVIRONMENT DETAILS FOUND for environment ID: ${environmentId}`);
      console.error(`ERROR: This environment ID does not exist in the environmentDetails table`);
      throw new Error(`No environment details found for environment ID: ${environmentId}`);
    }

    console.log(`SUCCESS: Environment details found for extracted ID ${environmentId}:`);
    console.log(`   - Host: ${environmentDetails.db_host}`);
    console.log(`   - Port: ${environmentDetails.port}`);
    console.log(`   - User: ${environmentDetails.db_user}`);
    console.log(`   - Database: ${environmentDetails.db_name}`);

    // Update DB_CONFIG with new environment details
    DB_CONFIG = {
      host: environmentDetails.db_host,
      port: parseInt(environmentDetails.port),
      user: environmentDetails.db_user,
      password: environmentDetails.db_password,
      database: environmentDetails.db_name
    };

    console.log(`SUCCESS: DB_CONFIG SUCCESSFULLY UPDATED FOR EXTRACTED ENVIRONMENT ${environmentId}`);
    console.log(`DEBUG: New DB_CONFIG for environment ${environmentId}:`, {
      host: DB_CONFIG.host,
      port: DB_CONFIG.port,
      user: DB_CONFIG.user,
      database: DB_CONFIG.database
      // Don't log password for security
    });
    
  } catch (err) {
    console.error(`ERROR: Failed to reinitialize DB_CONFIG for environment ${environmentId}:`, err);
    throw err;
  }
}
// --- MCP Server Setup ---
// const MCP_PORT = 8931;

// async function startMcpServer() {
//   const server = http.createServer((req, res) => {
//     if (req.url === '/health') {
//       res.writeHead(200, { 'Content-Type': 'text/plain' }).end('MCP Server OK');
//     } else {
//       res.writeHead(404).end();
//     }
//   });

//   const wss = new WebSocketServer({ server });
//   console.log(`MCP listening on ws://localhost:${MCP_PORT}`);

//   let browser: Browser | null = null;
//   let page: Page | null = null;
//   let sessionId: string | null = null;

//   wss.on('connection', ws => {
//     ws.on('message', async raw => {
//       let msg: any;
//       try { msg = JSON.parse(raw.toString()); }
//       catch { return; }

//       if (msg.jsonrpc !== '2.0' || typeof msg.id !== 'number') return;
//       const req: JsonRpcRequest = msg;

//       // initialize
//       if (req.method === 'initialize') {
//         sessionId = uuidv4();
//         if (browser) await browser.close();
//         browser = await chromium.launch({ headless: false });
//         page = await browser.newPage();

//         ws.send(JSON.stringify({
//           jsonrpc: '2.0',
//           id: req.id,
//           result: {
//             protocolVersion: '2025-03-26',
//             capabilities: {
//               tools: Object.fromEntries(
//                 Object.entries(tools).map(([k,tool]) => [k, { parameters: tool.parameters }])
//               )
//             },
//             serverInfo: { name: 'CustomPlaywrightMCP', version: '1.0.0', sessionId }
//           }
//         }));
//         ws.send(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }));
//         return;
//       }

//       // tools/call
//       if (req.method === 'tools/call') {
//         if (!page) {
//           ws.send(JSON.stringify({
//             jsonrpc: '2.0', id: req.id,
//             error: { code: -32000, message: 'Browser not initialized.' }
//           }));
//           return;
//         }
//         const { tool_name, tool_args } = req.params || {};
//         const tool = tools[tool_name];
//         if (!tool) {
//           ws.send(JSON.stringify({
//             jsonrpc: '2.0', id: req.id,
//             error: { code: -32601, message: `Tool "${tool_name}" not found.` }
//           }));
//           return;
//         }

//         try {
//           console.log(`→ ${tool_name}`, inspect(tool_args, false, 5, true));
//           const result = await tool.execute(
//             { page, browser: browser!, session_id: sessionId! },
//             tool_args
//           );
//           ws.send(JSON.stringify({ jsonrpc: '2.0', id: req.id, result }));
//         } catch (err: any) {
//           console.error(`Tool ${tool_name} error:`, err);
//           ws.send(JSON.stringify({
//             jsonrpc: '2.0', id: req.id,
//             error: { code: -32000, message: `Tool execution failed: ${err.message}` }
//           }));
//         }
//         return;
//       }

//       // unknown method
//       ws.send(JSON.stringify({
//         jsonrpc: '2.0', id: req.id,
//         error: { code: -32601, message: `Method "${req.method}" not found.` }
//       }));
//     });

//     ws.on('close', async () => {
//       if (browser) {
//         await browser.close();
//         browser = null;
//         page = null;
//         sessionId = null;
//       }
//     });
//   });

//   server.listen(MCP_PORT);
// }

// startMcpServer();





const MCP_PORT = 8931;

async function startMcpServer() {
  const server = http.createServer((req, res) => {
    if (req.url === '/health') {
      res.writeHead(200, { 'Content-Type': 'text/plain' }).end('MCP Server OK');
    } else {
      res.writeHead(404).end();
    }
  });

  wss = new WebSocketServer({ server });
  console.log(`MCP listening on ws://localhost:${MCP_PORT}`);

  let browser: Browser | null = null;
  let page: Page | null = null;
  let sessionId: string | null = null;

  wss.on('connection', ws => {
    console.debug('[DEBUG] Client connected');

    let isAuthenticated = false;
    let authenticatedAgentId: string | null = null;
    let jwtToken: string | null = null;
    
    ws.on('message', async raw => {
      let msg: any;
      try { 
        msg = JSON.parse(raw.toString()); 
        console.debug(`[DEBUG] Received: ${msg.method || 'unknown'} (id: ${msg.id || 'none'})`);
      }
      catch { return; }

      if (msg.method === 'authenticate') {
      const { agentId, secretKey } = msg.params || {};

      console.debug('[AUTH]  Authentication request received');
      console.debug('[AUTH]  Agent ID:', agentId);
      if (verifyStaticCredentials(agentId, secretKey)) {
      // Generate JWT token
      jwtToken = generateJWTToken(agentId);
      isAuthenticated = true;
      authenticatedAgentId = agentId;
      
      console.log('[AUTH]  Agent authenticated successfully:', agentId);
      
      ws.send(JSON.stringify({
        jsonrpc: '2.0',
        id: msg.id,
        result: {
          success: true,
          jwtToken: jwtToken,
          expiresIn: '24h',
          message: 'Authentication successful'
        }
      }));
    } else {
      console.error('[AUTH] Authentication failed for agent:', agentId);
      ws.send(JSON.stringify({
        jsonrpc: '2.0',
        id: msg.id,
        error: {
          code: 4001,
          message: 'Authentication failed: Invalid credentials'
        }
      }));
    }
    return;
  }
  // For all other requests, check JWT authentication
  if (!isAuthenticated) {
    console.error('[AUTH]  Request rejected: Agent not authenticated');
    ws.send(JSON.stringify({
      jsonrpc: '2.0',
      id: msg.id,
      error: {
        code: 4002,
        message: 'Authentication required: Please authenticate first'
      }
    }));
    return;
  }
  // Verify JWT token on each request
  const requestToken = msg.jwtToken; // JWT token extracted here
  if (!requestToken) {
    console.error('[JWT]  Request rejected: No JWT token provided');
    ws.send(JSON.stringify({
      jsonrpc: '2.0',
      id: msg.id,
      error: {
        code: 4003,
        message: 'JWT token required for this request'
      }
    }));
    return;
  }
  const tokenVerification = verifyJWTToken(requestToken);
  if (!tokenVerification.valid || tokenVerification.agentId !== authenticatedAgentId) {
    console.error('[JWT] Request rejected: Invalid JWT token');
    ws.send(JSON.stringify({
      jsonrpc: '2.0',
      id: msg.id,
      error: {
        code: 4004,
        message: 'Invalid JWT token',
        data: {
          newToken: PERSISTENT_JWT_TOKEN,
          expiresIn: `${JWT_REFRESH_INTERVAL_MINUTES}m`
        }
      }
    }));
    return;
  }
  
  if (tokenVerification.needsRefresh) {
    console.log('[JWT] Client using previous token - refresh recommended');
  }
      // Check if this is a WebSocket server message (your functionality)
      if (msg.method && ['DBSchema', 'execute_query', 'echo', 'notifications/initialized'].includes(msg.method)) {
        await handleWebSocketMessage(ws, msg);
        return;
      }

      // Original MCP server logic
      if (msg.jsonrpc !== '2.0' || typeof msg.id !== 'number') return;
      const req: JsonRpcRequest = msg;

      // initialize
      if (req.method === 'initialize') {
        sessionId = uuidv4();
        if (browser) await browser.close();
        browser = await chromium.launch({ headless: false });
        page = await browser.newPage();

        ws.send(JSON.stringify({
          jsonrpc: '2.0',
          id: req.id,
          result: {
            protocolVersion: '2025-03-26',
            capabilities: {
              tools: Object.fromEntries(
                Object.entries(tools).map(([k,tool]) => [k, { parameters: tool.parameters }])
              )
            },
            serverInfo: { name: 'CustomPlaywrightMCP', version: '1.0.0', sessionId }
          }
        }));
        ws.send(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }));
        return;
      }

      // tools/call
      if (req.method === 'tools/call') {
        if (!page) {
          ws.send(JSON.stringify({
            jsonrpc: '2.0', id: req.id,
            error: { code: -32000, message: 'Browser not initialized.' }
          }));
          return;
        }
        const { tool_name, tool_args } = req.params || {};
        const tool = tools[tool_name];
        if (!tool) {
          ws.send(JSON.stringify({
            jsonrpc: '2.0', id: req.id,
            error: { code: -32601, message: `Tool "${tool_name}" not found.` }
          }));
          return;
        }

        try {
          console.log(`→ ${tool_name}`, inspect(tool_args, false, 5, true));
          const result = await tool.execute(
            { page, browser: browser!, session_id: sessionId! },
            tool_args
          );
          ws.send(JSON.stringify({ jsonrpc: '2.0', id: req.id, result }));
        } catch (err: any) {
          console.error(`Tool ${tool_name} error:`, err);
          ws.send(JSON.stringify({
            jsonrpc: '2.0', id: req.id,
            error: { code: -32000, message: `Tool execution failed: ${err.message}` }
          }));
        }
        return;
      }

      // unknown method
      ws.send(JSON.stringify({
        jsonrpc: '2.0', id: req.id,
        error: { code: -32601, message: `Method "${req.method}" not found.` }
      }));
    });

    ws.on('close', async () => {
      console.debug('[DEBUG] Client disconnected');
      if (browser) {
        await browser.close();
        browser = null;
        page = null;
        sessionId = null;
      }
      console.debug(`[JWT] Connection closed but persistent token remains active`);
    });

    ws.on('error', err => {
      console.error('[WebSocket Error]', err);
    });
  });

  server.listen(MCP_PORT);
}

// Test database connection before starting server
(async () => {
    try {

        // Wait for DB_CONFIG to be initialized
        
        // Load RDS credentials from API
        console.log('[STARTUP] Loading RDS credentials from API...');
        const rdsCredentials = await getRDSCredentialsFromAPI();

        initialDbConfig = isProduction
            ? {
                host: rdsCredentials.rdsHost,
                user: rdsCredentials.rdsUser,
                password: rdsCredentials.rdsPass,
                database: rdsCredentials.rdsDb,
                waitForConnections: true,
                connectionLimit: 15,
                queueLimit: 45,
                enableKeepAlive: true,
                keepAliveInitialDelay: 0,
                multipleStatements: true,
            }
           : {
                host: 'localhost',
                port: 3306,
                user: process.env.LOCAL_USER!,
                password: process.env.LOCAL_PASS!,
                database: process.env.LOCAL_DB!,
                waitForConnections: true,
                connectTimeout: 10000,
                enableKeepAlive: true,
                multipleStatements: true,
            };

        // Create the initial database pool
        initialDbPool = mysql.createPool(initialDbConfig);
        console.log('[STARTUP] RDS credentials loaded and pool created successfully');


        // Now DB_CONFIG is guaranteed to be available
        const credentials = await getCredentialsFromAPI();
        VALID_AGENT_ID = credentials.agentId;
        VALID_AGENT_SECRET = credentials.secretKey;

        if (!VALID_AGENT_ID || !VALID_AGENT_SECRET) {
            throw new Error('Failed to load valid credentials from AWS Secrets');
        }
        
        console.log('[STARTUP]  Credentials loaded successfully:');
        console.log(`[STARTUP]    Agent ID: "${VALID_AGENT_ID}"`);
        console.log(`[STARTUP]    Secret Key: "${VALID_AGENT_SECRET}"`);
        console.log(`[STARTUP]    Secret Key Length: ${VALID_AGENT_SECRET.length} characters`);
        console.log(`[STARTUP]    Agent ID: "${VALID_AGENT_ID}"`);
        console.log(`[STARTUP]    Secret Key: "${VALID_AGENT_SECRET}"`);

        initializePersistentToken();

        console.log('[STARTUP] Testing database connection...');
        const conn = await mysql.createConnection(initialDbConfig);
        await conn.end();
        console.log('[STARTUP] Initial database connection test successful');

        console.log('[STARTUP] Note: Target environment database will be configured when agent connects');
        console.debug('[DEBUG] Database connection test successful');
        // Assuming MCP_PORT and startMcpServer() are defined elsewhere
        console.log(`Starting integrated server with WebSocket MCP and TDGen on ws://localhost:${MCP_PORT}`);
        await startMcpServer();
    } catch (e) {
        console.error('[ERROR] Database connection failed:', e);
        // Start server anyway, but WebSocket functionality may be limited
        await startMcpServer();
    }
})();



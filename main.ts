// free-model-aggregator-01: one server, two client wires, a directory of free providers.
//
// The server reads the brief, probes each provider's catalog route in the provider's own wire, builds
// one catalog, then:
//   1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//   2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//   3. serves the client's requests: GET /v1/models (answering in the shape of the auth header it got:
//      x-api-key means Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST
//      /v1/messages, each with a JSON body or with server-sent events;
//   4. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//      process still running when the wall ends scores zero, so never wait for anything else.
// `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the register
// body's catalog must be that same document. Build it once, as this file does, and use the one object.
//
// Routing: for each request, every free model that meets its requirements is ranked by the quality
// table's rating_milli, best first, one model a provider at a time (each provider's best model before
// any provider's second-best). The first pick is the best-ranked model on a provider this trial has
// called fewer than PLAN_CAP times, so the load spreads over the free tiers instead of draining the
// best provider dry; a provider that refuses (402, 429, 5xx, a timeout) is counted and the request is
// retried on the next ranked model, until one answers or the list runs out. One exception: the
// best-rated model gets a single immediate retry when it refuses, because a refusal on its schedule is
// usually one bad call number, not a spent budget, and recovering the best model is worth more than
// the call it costs. A provider that answered
// 2xx is never called twice for one request: the scorer counts a request answered by two providers as
// invalid. Every choice is a function of the brief, the requests and the providers' answers so far:
// no clock, no unseeded randomness, so a replay leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  candidatesFor,
  isRecord,
  joinBase,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type Candidate,
  type CatalogReport,
  type Json,
  type Probe,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromGemini,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAIResponse,
  geminiToAnthropicResponse,
  geminiToOpenAIResponse,
  openAIError,
  openAIEventsFromAnthropic,
  openAIEventsFromGemini,
  openAIModelList,
  openAIToAnthropic,
  toAnthropicRequest,
  toGeminiRequest,
  toOpenAIRequest,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds (100 ms). */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider inference call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 10000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** The most provider attempts for one client request. */
const MAX_ATTEMPTS = 15;
/**
 * How many calls one provider takes before the first pick moves on: every provider's per-trial budget
 * is at least a few calls, so a handful of planned calls can never drain one, and the spread keeps an
 * outage or a spent budget from taking many requests down with it.
 */
const PLAN_CAP = 4;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Every provider call made so far in this trial, failed ones included: the load-spreading ledger. */
const usedCalls = new Map<string, number>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of probe.models) {
      if (seen.has(model.id)) continue;
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(clientApi: 'openai' | 'anthropic', body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for a
  // brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** The upstream request for one candidate: its URL, headers and body, in the provider's own wire. */
function upstreamRequest(candidate: Candidate, clientApi: 'openai' | 'anthropic', body: Json): { url: string; headers: Record<string, string>; body: string } {
  const base = joinBase(ENV_URL, candidate.probe.provider.base);
  const provider = candidate.probe.provider;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    return { url: `${base}/v1/chat/completions`, headers, body: JSON.stringify(toOpenAIRequest(clientApi, body, candidate.model.id)) };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    return { url: `${base}/v1/messages`, headers, body: JSON.stringify(toAnthropicRequest(clientApi, body, candidate.model.id)) };
  }
  const gemini = toGeminiRequest(clientApi, body, candidate.model.id, body.stream === true);
  return { url: `${base}${gemini.path}`, headers, body: JSON.stringify(gemini.body) };
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/**
 * One upstream JSON answer translated into the client's wire. Same-wire answers pass through byte for
 * byte, so their usage counts arrive exactly as the provider reported them; cross-wire answers are
 * rebuilt from the provider's text, stop reason and usage.
 */
async function answerJson(clientApi: 'openai' | 'anthropic', candidate: Candidate, upstream: Response): Promise<{ status: number; contentType: string; text: string }> {
  const text = await upstream.text();
  const wire = candidate.probe.provider.wire;
  if (clientApi === 'openai' && wire === 'openai') return { status: 200, contentType: 'application/json', text };
  if (clientApi === 'anthropic' && wire === 'anthropic') return { status: 200, contentType: 'application/json', text };
  let parsed: unknown = null;
  try {
    parsed = JSON.parse(text);
  } catch {
    parsed = null;
  }
  const body: Json = isRecord(parsed) ? parsed : {};
  let out: Json;
  if (clientApi === 'openai' && wire === 'anthropic') out = anthropicToOpenAIResponse(body, candidate.model.id);
  else if (clientApi === 'openai' && wire === 'gemini') out = geminiToOpenAIResponse(body, candidate.model.id);
  else if (clientApi === 'anthropic' && wire === 'openai') out = openAIToAnthropic(body, candidate.model.id);
  else out = geminiToAnthropicResponse(body, candidate.model.id);
  return { status: 200, contentType: 'application/json', text: JSON.stringify(out) };
}

/**
 * One upstream SSE answer translated into the client's wire. Same-wire streams pass through byte for
 * byte; cross-wire streams are re-emitted event by event, in the client's order, with the provider's
 * text, stop reason and usage carried across.
 */
async function answerStream(clientApi: 'openai' | 'anthropic', candidate: Candidate, upstream: Response, res: ServerResponse): Promise<void> {
  const wire = candidate.probe.provider.wire;
  res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
  if (upstream.body === null) {
    res.end('data: [DONE]\n\n');
    return;
  }
  if (clientApi === 'openai' && wire === 'openai') {
    await pipeStream(upstream.body, res);
    return;
  }
  if (clientApi === 'anthropic' && wire === 'anthropic') {
    await pipeStream(upstream.body, res);
    return;
  }
  const model = candidate.model.id;
  if (clientApi === 'openai' && wire === 'anthropic') {
    for await (const event of openAIEventsFromAnthropic(upstream.body, model)) res.write(event);
  } else if (clientApi === 'openai' && wire === 'gemini') {
    for await (const event of openAIEventsFromGemini(upstream.body, model)) res.write(event);
  } else if (clientApi === 'anthropic' && wire === 'openai') {
    for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
  } else {
    for await (const event of anthropicEventsFromGemini(upstream.body, model)) res.write(event);
  }
  res.end();
}

/**
 * Serve one client request: walk the ranked candidates, first pick under the plan cap, then the rest in
 * rating order, until one provider answers 2xx. A 2xx answer is the request's answer even if its body
 * surprises: calling a second provider after one answered would make the request invalid.
 */
async function serve(clientApi: 'openai' | 'anthropic', req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, clientApi, 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(clientApi, body);
  const candidates = candidatesFor(probes, brief.ratings, clientApi, requires);
  if (candidates.length === 0) {
    return sendError(res, clientApi, 503, 'no_free_model', 'no free model on the directory meets this request');
  }
  const stream = body.stream === true;
  // The first pick is the best-ranked model on a provider under the plan cap; when every provider is
  // at the cap the best-ranked model serves anyway.
  const first = candidates.find((c) => (usedCalls.get(c.provider.id) ?? 0) < PLAN_CAP) ?? candidates[0];
  const ordered = [first, ...candidates.filter((c) => c !== first)];
  const maxRating = candidates[0].rating;
  const tried = new Set<string>();
  const retried = new Set<string>();
  let attempts = 0;
  let lastStatus = 502;
  let lastDetail = 'every provider refused the request';
  // A refusal of the best-rated model earns one immediate retry of that same model: a scheduled
  // refusal is one bad call number, and the next call number usually answers.
  const retrySame = (candidate: Candidate, key: string): boolean => {
    if (candidate.rating !== maxRating || retried.has(key)) return false;
    retried.add(key);
    tried.delete(key);
    return true;
  };
  let index = 0;
  while (index < ordered.length && attempts < MAX_ATTEMPTS) {
    const candidate = ordered[index];
    const key = `${candidate.provider.id}/${candidate.model.id}`;
    if (tried.has(key)) {
      index += 1;
      continue;
    }
    tried.add(key);
    attempts += 1;
    usedCalls.set(candidate.provider.id, (usedCalls.get(candidate.provider.id) ?? 0) + 1);
    let upstream: Response;
    try {
      const request = upstreamRequest(candidate, clientApi, body);
      upstream = await fetch(request.url, {
        method: 'POST',
        headers: request.headers,
        body: request.body,
        signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
      });
    } catch (err) {
      lastStatus = 502;
      lastDetail = `the ${candidate.provider.id} provider could not be reached: ${(err as Error).message}`;
      log(lastDetail);
      if (retrySame(candidate, key)) continue;
      index += 1;
      continue;
    }
    if (!upstream.ok) {
      lastStatus = upstream.status;
      let detail = '';
      try {
        detail = (await upstream.text()).slice(0, 300);
      } catch {
        detail = '';
      }
      lastDetail = `the ${candidate.provider.id} provider answered HTTP ${upstream.status}${detail !== '' ? `: ${detail}` : ''}`;
      log(lastDetail);
      if (retrySame(candidate, key)) continue;
      index += 1;
      continue;
    }
    // A 2xx answer ends the request: translate it into the client's wire and send it.
    try {
      if (stream) {
        await answerStream(clientApi, candidate, upstream, res);
      } else {
        const answer = await answerJson(clientApi, candidate, upstream);
        res.writeHead(answer.status, { 'content-type': answer.contentType, 'content-length': Buffer.byteLength(answer.text) });
        res.end(answer.text);
      }
    } catch (err) {
      log(`translating the ${candidate.provider.id} answer failed: ${(err as Error).message}`);
      if (!res.headersSent) return sendError(res, clientApi, 502, 'upstream_error', 'the provider answer could not be translated');
      res.end();
    }
    return;
  }
  return sendError(res, clientApi, lastStatus >= 400 && lastStatus < 600 ? lastStatus : 502, 'upstream_error', lastDetail);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await serve('openai', req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await serve('anthropic', req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  let lastError: Error | null = null;
  for (let attempt = 1; attempt <= 5; attempt++) {
    try {
      const res = await fetch(`${ENV_URL}/register`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
        signal: AbortSignal.timeout(5000),
      });
      if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
      const reply: unknown = await res.json();
      if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
      return;
    } catch (err) {
      lastError = err as Error;
      log(`register attempt ${attempt} failed: ${lastError.message}`);
      if (attempt < 5) await new Promise((resolve) => setTimeout(resolve, 200));
    }
  }
  throw lastError ?? new Error('register failed');
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}

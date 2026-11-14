// The client wires and the three upstream wires.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. The entry forwards each request to a
// provider on that provider's own wire and translates the answer back:
//
//   - openai upstream: the Chat Completions body passes through (with the routed model's id); a stream
//     always carries stream_options.include_usage, because the client never asks for it and the usage
//     check needs the provider's usage chunk.
//   - anthropic upstream: the request is rewritten to Messages (system prompt, image blocks, tool
//     definitions, tool results and tool choice included) and the answer back; a JSON-mode request never
//     goes here, because the Messages wire cannot carry response_format and its echo would say so.
//   - gemini upstream: the request is rewritten to generateContent / streamGenerateContent (system
//     instruction, contents with inlineData images, functionDeclarations, generationConfig with
//     maxOutputTokens, stopSequences and responseMimeType) and the answer back.
//
// The provider echoes what it parsed, and the client compares that echo with what it sent: every
// translation preserves the system text, the user text, the tool names in order, max_tokens, the stop
// sequences, the image bytes and the JSON mode, so the echo matches on whichever wire served it.
//
// Everything here is a pure function of its arguments: no clock and no randomness, so the same body and
// the same provider answer give the same bytes on a replay. The two model listings use fixed timestamps
// for the same reason: a `created` that moved between runs would be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** The text of every system message in an OpenAI body, joined the way the provider's parser joins them. */
function openAISystemText(body: Json): string {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  const parts: string[] = [];
  for (const message of messages) {
    if (!isRecord(message) || message.role !== 'system') continue;
    const text = textOf(message.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** A data: URL split into its media type and base64 payload; null when it is not one. */
function dataUrlParts(url: string): { media: string; data: string } | null {
  const match = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (match === null) return null;
  return { media: match[1] !== '' ? match[1] : 'image/png', data: match[2] };
}

/** One OpenAI image_url part as an Anthropic image block. */
function openAIImageToAnthropic(imageUrl: Json): Json | null {
  const url = typeof imageUrl.url === 'string' ? imageUrl.url : '';
  if (url === '') return null;
  const data = dataUrlParts(url);
  if (data !== null) return { type: 'image', source: { type: 'base64', media_type: data.media, data: data.data } };
  return { type: 'image', source: { type: 'url', url } };
}

/** One OpenAI content part as an Anthropic content block: text or an inline image; null for the rest. */
function openAIPartToAnthropic(part: Json): Json | null {
  if (part.type === 'text' && typeof part.text === 'string') return { type: 'text', text: part.text };
  if (part.type === 'image_url' && isRecord(part.image_url)) return openAIImageToAnthropic(part.image_url);
  return null;
}

/** A tool_result's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic's stop_reason as OpenAI's finish_reason. */
function finishReasonOf(stop: unknown): string {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as OpenAI's finish_reason. */
function finishReasonOfGemini(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as Anthropic's stop_reason. */
function stopReasonOfGemini(reason: unknown): string {
  if (reason === 'MAX_TOKENS') return 'max_tokens';
  if (reason === 'SAFETY') return 'refusal';
  return 'end_turn';
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Client request -> upstream request
// ---------------------------------------------------------------------------

/**
 * A client request as an OpenAI Chat Completions request that names `model`. An OpenAI client passes
 * through (with the routed model and, for a stream, stream_options.include_usage forced on, so the
 * provider's usage chunk arrives); an Anthropic client is rewritten: the system prompt becomes a system
 * message, image blocks become image_url data URLs, tool_use blocks become tool_calls, tool_result
 * blocks become tool messages, tools and tool_choice are rewritten, and the output cap, temperature,
 * top_p, stop sequences and streaming pass through.
 */
export function toOpenAIRequest(clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  if (clientApi === 'openai') {
    const out: Json = { ...body, model };
    if (body.stream === true) {
      const options = isRecord(body.stream_options) ? { ...body.stream_options } : {};
      options.include_usage = true;
      out.stream_options = options;
    }
    return out;
  }
  return anthropicToOpenAI(body, model);
}

/** An Anthropic Messages request as an OpenAI Chat Completions request that names `model`. */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/**
 * A client request as an Anthropic Messages request that names `model`. An Anthropic client passes
 * through (with the routed model); an OpenAI client is rewritten: system messages become the system
 * prompt, user parts become text and image blocks, assistant tool_calls become tool_use blocks, tool
 * messages become tool_result blocks, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through. Never called for a JSON-mode OpenAI
 * request: the Messages wire cannot carry response_format.
 */
export function toAnthropicRequest(clientApi: 'openai' | 'anthropic', body: Json, model: string): Json {
  if (clientApi === 'anthropic') return { ...body, model };
  const messages: Json[] = [];
  const system = openAISystemText(body);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') continue;
    if (role === 'assistant') {
      const blocks: Json[] = [];
      const content = raw.content;
      if (typeof content === 'string') {
        if (content !== '') blocks.push({ type: 'text', text: content });
      } else if (Array.isArray(content)) {
        for (const part of content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
        }
      }
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        blocks.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: toolResultText(raw.content) }],
      });
      continue;
    }
    // A user message: text and image parts become Anthropic blocks, in order.
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      const block = openAIPartToAnthropic(part);
      if (block !== null) blocks.push(block);
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (system !== '') out.system = system;
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop;
  else if (typeof body.stop === 'string' && body.stop !== '') out.stop_sequences = [body.stop];
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [
        {
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (typeof choice === 'string') {
    if (choice === 'auto') out.tool_choice = { type: 'auto' };
    else if (choice === 'none') out.tool_choice = { type: 'none' };
    else if (choice === 'required') out.tool_choice = { type: 'any' };
  } else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One client message's text and image parts as Gemini content parts. */
function geminiPartsOfOpenAI(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const data = dataUrlParts(url);
      if (data !== null) parts.push({ inlineData: { mimeType: data.media, data: data.data } });
      else parts.push({ fileData: { fileUri: url } });
    }
  }
  return parts;
}

/** One Anthropic user message's blocks as Gemini content parts (tool_result blocks are dropped). */
function geminiPartsOfAnthropic(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      const source = block.source;
      if (source.type === 'base64' && typeof source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof source.media_type === 'string' ? source.media_type : 'image/png', data: source.data } });
      } else if (source.type === 'url' && typeof source.url === 'string') {
        parts.push({ fileData: { fileUri: source.url } });
      }
    }
  }
  return parts;
}

/**
 * A client request as a Gemini generateContent body, with the path it is POSTed to. The model travels in
 * the path, not the body. System text becomes the systemInstruction, user messages become user contents
 * (text and inlineData images), assistant messages become model contents (text and functionCall parts),
 * tool messages are dropped (they carry no text the provider echoes), tools become functionDeclarations
 * and the output cap, stop sequences, temperature, top_p and JSON mode become the generationConfig.
 */
export function toGeminiRequest(clientApi: 'openai' | 'anthropic', body: Json, model: string, stream: boolean): { path: string; body: Json } {
  const systemText = clientApi === 'openai' ? openAISystemText(body) : textOf(body.system);
  const contents: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (clientApi === 'openai') {
        parts.push(...geminiPartsOfOpenAI(raw.content));
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
        }
      } else {
        if (Array.isArray(raw.content)) {
          for (const block of raw.content) {
            if (!isRecord(block)) continue;
            if (block.type === 'text' && typeof block.text === 'string') parts.push({ text: block.text });
            else if (block.type === 'tool_use') {
              parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
            }
          }
        } else if (typeof raw.content === 'string' && raw.content !== '') {
          parts.push({ text: raw.content });
        }
      }
      contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') continue;
    // A user message. Its parts are kept even when empty, so the provider's "last user content" is this
    // message, as the client's parser reads it.
    const parts = clientApi === 'openai' ? geminiPartsOfOpenAI(raw.content) : geminiPartsOfAnthropic(raw.content);
    contents.push({ role: 'user', parts });
  }

  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  const declarations: Json[] = [];
  if (Array.isArray(body.tools)) {
    for (const raw of body.tools) {
      if (clientApi === 'openai') {
        if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
        declarations.push({
          name: raw.function.name,
          description: typeof raw.function.description === 'string' ? raw.function.description : '',
          parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
        });
      } else {
        if (!isRecord(raw) || typeof raw.name !== 'string') continue;
        declarations.push({
          name: raw.name,
          description: typeof raw.description === 'string' ? raw.description : '',
          parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
        });
      }
    }
  }
  if (declarations.length > 0) out.tools = [{ functionDeclarations: declarations }];
  const config: Json = {};
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : undefined;
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  const stops = clientApi === 'openai' ? body.stop : body.stop_sequences;
  if (typeof stops === 'string' && stops !== '') config.stopSequences = [stops];
  else if (Array.isArray(stops) && stops.length > 0) config.stopSequences = stops;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (clientApi === 'openai' && isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  const path = `/v1beta/models/${encodeURIComponent(model)}:${stream ? 'streamGenerateContent' : 'generateContent'}`;
  return { path, body: out };
}

// ---------------------------------------------------------------------------
// Upstream response -> client response (JSON)
// ---------------------------------------------------------------------------

/** A Chat Completions answer as an Anthropic Messages object. `model` is the serving model id. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** An Anthropic Messages answer as a Chat Completions object. `model` is the serving model id. */
export function anthropicToOpenAIResponse(body: Json, model: string): Json {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('');
  const toolCalls: Json[] = [];
  for (const block of content) {
    if (!isRecord(block) || block.type !== 'tool_use') continue;
    toolCalls.push({
      id: typeof block.id === 'string' ? block.id : 'call_unknown',
      type: 'function',
      function: { name: typeof block.name === 'string' ? block.name : '', arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) },
    });
  }
  const usage = isRecord(body.usage) ? body.usage : {};
  const message: Json = { role: 'assistant', content: text };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: finishReasonOf(body.stop_reason), logprobs: null }],
    usage: {
      prompt_tokens: numberOrZero(usage.input_tokens),
      completion_tokens: numberOrZero(usage.output_tokens),
      total_tokens: numberOrZero(usage.input_tokens) + numberOrZero(usage.output_tokens),
    },
  };
}

/** The text of a Gemini candidate's content parts, joined. */
function geminiTextOf(body: Json): string {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(first.content) ? first.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  return parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
}

/** A Gemini candidate's finishReason, or null. */
function geminiFinishOf(body: Json): string | null {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  return typeof first.finishReason === 'string' ? first.finishReason : null;
}

/** A Gemini answer's usageMetadata as prompt/output/total token counts. */
function geminiUsageOf(body: Json): { input: number; output: number; total: number } {
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const input = numberOrZero(meta.promptTokenCount);
  const output = numberOrZero(meta.candidatesTokenCount);
  return { input, output, total: numberOrZero(meta.totalTokenCount) || input + output };
}

/** A Gemini generateContent answer as a Chat Completions object. `model` is the serving model id. */
export function geminiToOpenAIResponse(body: Json, model: string): Json {
  const usage = geminiUsageOf(body);
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, message: { role: 'assistant', content: geminiTextOf(body) }, finish_reason: finishReasonOfGemini(geminiFinishOf(body)), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.total },
  };
}

/** A Gemini generateContent answer as an Anthropic Messages object. `model` is the serving model id. */
export function geminiToAnthropicResponse(body: Json, model: string): Json {
  const usage = geminiUsageOf(body);
  return {
    id: 'msg_sim',
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: geminiTextOf(body) }],
    stop_reason: stopReasonOfGemini(geminiFinishOf(body)),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

// ---------------------------------------------------------------------------
// SSE helpers
// ---------------------------------------------------------------------------

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** One parsed SSE frame: its event name (null when unnamed) and its data payload lines joined. */
export interface SseFrame {
  event: string | null;
  data: string;
}

/** A stream of bytes as SSE frames, split on blank lines; `\r\n` is normalised away. */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        yield parseSseFrame(frame);
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') yield parseSseFrame(buffer);
  } finally {
    reader.releaseLock();
  }
}

function parseSseFrame(frame: string): SseFrame {
  let event: string | null = null;
  const data: string[] = [];
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
    else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
  }
  return { event, data: data.join('\n') };
}

/** One OpenAI chat chunk object. */
function openAIChunk(model: string, delta: Json, finish: string | null): Json {
  return { id: 'chatcmpl-sim', object: 'chat.completion.chunk', created: FIXED_OPENAI_CREATED, model, choices: [{ index: 0, delta, finish_reason: finish }] };
}

/** One OpenAI chat usage chunk (no choices), as the provider sends it when include_usage is on. */
function openAIUsageChunk(model: string, input: number, output: number): Json {
  return {
    id: 'chatcmpl-sim',
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [],
    usage: { prompt_tokens: input, completion_tokens: output, total_tokens: input + output },
  };
}

// ---------------------------------------------------------------------------
// Upstream response -> client response (SSE)
// ---------------------------------------------------------------------------

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the serving model id.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's payload as Anthropic events, mutating the stream's state. */
  const eventsOfPayload = (payload: string): string[] => {
    const out: string[] = [];
    if (payload === '[DONE]') {
      out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
      return out;
    }
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      return out;
    }
    if (!isRecord(chunk)) return out;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return out;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    out.push(...startEvents());
    if (typeof delta.content === 'string' && delta.content !== '') {
      if (textIndex === -1) {
        textIndex = nextIndex;
        nextIndex += 1;
        out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
      }
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
    }
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
      let index = toolIndexes.get(upstreamIndex);
      if (index === undefined) {
        if (textIndex !== -1) {
          out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
          textIndex = -1;
        }
        index = nextIndex;
        nextIndex += 1;
        toolIndexes.set(upstreamIndex, index);
        const fn = isRecord(call.function) ? call.function : {};
        out.push(
          sseEvent('content_block_start', {
            type: 'content_block_start',
            index,
            content_block: {
              type: 'tool_use',
              id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
              name: typeof fn.name === 'string' ? fn.name : '',
              input: {},
            },
          }),
        );
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.arguments === 'string' && fn.arguments !== '') {
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
      }
    }
    // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
    // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    return out;
  };

  for await (const frame of sseFrames(upstream)) {
    for (const event of eventsOfPayload(frame.data.trim())) yield event;
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

/**
 * An Anthropic SSE stream as OpenAI chat chunks: a role chunk on message_start, content deltas for text
 * and tool-call arguments, a finish chunk on message_delta and a usage chunk, then [DONE]. `model` is
 * the serving model id.
 */
export async function* openAIEventsFromAnthropic(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let roleSent = false;
  let finished = false;
  let inputTokens = 0;
  let outputTokens = 0;
  let pendingStop: string | null = null;
  const toolBlocks = new Map<number, { id: string; name: string; sentHead: boolean }>();

  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    const out = [sseEvent(null, openAIChunk(model, {}, pendingStop === null ? 'stop' : finishReasonOf(pendingStop)))];
    out.push(sseEvent(null, openAIUsageChunk(model, inputTokens, outputTokens)));
    out.push('data: [DONE]\n\n');
    return out;
  };

  for await (const frame of sseFrames(upstream)) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    switch (frame.event) {
      case 'message_start': {
        if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
        if (!roleSent) {
          roleSent = true;
          yield sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null));
        }
        break;
      }
      case 'content_block_start': {
        const index = typeof data.index === 'number' ? data.index : -1;
        const block = isRecord(data.content_block) ? data.content_block : {};
        if (block.type === 'tool_use' && index >= 0) {
          toolBlocks.set(index, {
            id: typeof block.id === 'string' ? block.id : `call_${index}`,
            name: typeof block.name === 'string' ? block.name : '',
            sentHead: false,
          });
        }
        break;
      }
      case 'content_block_delta': {
        const index = typeof data.index === 'number' ? data.index : -1;
        const delta = isRecord(data.delta) ? data.delta : {};
        if (delta.type === 'text_delta' && typeof delta.text === 'string' && delta.text !== '') {
          if (!roleSent) {
            roleSent = true;
            yield sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null));
          }
          yield sseEvent(null, openAIChunk(model, { content: delta.text }, null));
        } else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string' && delta.partial_json !== '') {
          const block = toolBlocks.get(index);
          if (block !== undefined) {
            if (!block.sentHead) {
              block.sentHead = true;
              yield sseEvent(null, openAIChunk(model, { tool_calls: [{ index, id: block.id, type: 'function', function: { name: block.name, arguments: '' } }] }, null));
            }
            yield sseEvent(null, openAIChunk(model, { tool_calls: [{ index, function: { arguments: delta.partial_json } }] }, null));
          }
        }
        break;
      }
      case 'message_delta': {
        if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') pendingStop = data.delta.stop_reason;
        if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
        break;
      }
      case 'message_stop': {
        for (const event of finishEvents()) yield event;
        break;
      }
      default:
        break;
    }
  }
  if (!finished) for (const event of finishEvents()) yield event;
}

/**
 * A Gemini streamGenerateContent SSE stream as OpenAI chat chunks: a role chunk, content deltas for the
 * text pieces, a finish chunk and a usage chunk from the final frame's usageMetadata, then [DONE].
 * `model` is the serving model id.
 */
export async function* openAIEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let roleSent = false;
  let finished = false;
  let pendingFinish: string | null = null;
  let inputTokens = 0;
  let outputTokens = 0;

  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    return [
      sseEvent(null, openAIChunk(model, {}, finishReasonOfGemini(pendingFinish ?? 'STOP'))),
      sseEvent(null, openAIUsageChunk(model, inputTokens, outputTokens)),
      'data: [DONE]\n\n',
    ];
  };

  for await (const frame of sseFrames(upstream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    const usage = geminiUsageOf(data);
    if (isRecord(data.usageMetadata)) {
      inputTokens = usage.input;
      outputTokens = usage.output;
    }
    const text = geminiTextOf(data);
    if (text !== '') {
      if (!roleSent) {
        roleSent = true;
        yield sseEvent(null, openAIChunk(model, { role: 'assistant', content: '' }, null));
      }
      yield sseEvent(null, openAIChunk(model, { content: text }, null));
    }
    const finish = geminiFinishOf(data);
    if (finish !== null) pendingFinish = finish;
  }
  for (const event of finishEvents()) yield event;
}

/**
 * A Gemini streamGenerateContent SSE stream as the Anthropic event sequence: message_start, one text
 * content block with its deltas, message_delta with the stop reason and the output tokens, then
 * message_stop. `model` is the serving model id.
 */
export async function* anthropicEventsFromGemini(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let started = false;
  let blockOpen = false;
  let finished = false;
  let pendingStop: string | null = null;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    blockOpen = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: { id: 'msg_stream', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: 0, output_tokens: 0 } },
      }),
      sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
    ];
  };
  const finishEvents = (): string[] => {
    if (finished) return [];
    finished = true;
    const out: string[] = [];
    if (blockOpen) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }));
      blockOpen = false;
    }
    out.push(
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: pendingStop ?? 'end_turn', stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    );
    return out;
  };

  for await (const frame of sseFrames(upstream)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (isRecord(data.usageMetadata)) outputTokens = geminiUsageOf(data).output;
    const text = geminiTextOf(data);
    if (text !== '') {
      for (const event of startEvents()) yield event;
      yield sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } });
    }
    const finish = geminiFinishOf(data);
    if (finish !== null) pendingStop = stopReasonOfGemini(finish);
  }
  if (!started) for (const event of startEvents()) yield event;
  for (const event of finishEvents()) yield event;
}

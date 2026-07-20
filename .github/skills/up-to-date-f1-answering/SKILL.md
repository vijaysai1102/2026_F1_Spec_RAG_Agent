---
name: Up-to-Date F1 Answering
description: Use this skill when a question asks for current, latest, recent, or as-of-now Formula 1 information.
---

# Up-to-Date F1 Answering

## Purpose
This skill improves answer quality for time-sensitive questions by combining:
1. The repository's 2026 F1 regulations RAG context.
2. Fresh web sources for current updates.

Use this skill when the user asks for:
- latest/current/recent updates
- "as of now" facts
- this season, this week, or this year changes

## Required behavior
1. Detect if the query is time-sensitive.
2. Retrieve relevant live web sources from authoritative sites (FIA, Formula1.com, teams, official announcements, reliable motorsport outlets).
3. Merge live findings with regulation context from the repository.
4. Clearly separate:
   - **Regulation-backed claims** (cite article/page when possible)
   - **Live-update claims** (cite URL + date)
5. If live information is unavailable, state that explicitly and answer with best-known regulation context only.

## Output format
Return responses with:
1. **Answer**
2. **Evidence**
   - Regulation citations (Article/page)
   - Live sources (URL list)
3. **Freshness note** (UTC timestamp used for live lookup)

## Guardrails
- Never present live claims without source URLs.
- Never claim a regulation article if not present in retrieved context.
- If sources conflict, note the conflict and prefer official sources.

## Example trigger prompts
- "What is the latest update on 2026 power unit rules?"
- "As of now, what are the active aero mode names?"
- "Any recent FIA clarification about 2026 chassis dimensions?"

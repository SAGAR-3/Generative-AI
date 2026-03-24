"""
LLM Service
Orchestrates prompt construction + LLM call (Anthropic Claude).
Returns structured TestCase objects.
"""
from __future__ import annotations
import json
import os
import re
from models.schemas import TestCase, TestType, Priority, TestStep

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a senior QA engineer and test architect. 
Your job is to generate comprehensive, precise, and traceable test cases from software Business Requirement Documents (BRDs).

For each requirement or user story you analyse, generate test cases covering:
- Functional (happy path + alternate flows)
- Edge cases (boundary values, limits, empty inputs)
- Negative cases (invalid data, error handling)
- Non-functional (performance thresholds, security checks)

Each test case MUST follow this exact JSON schema:
{
  "title": "Short descriptive title",
  "test_type": "functional|edge|negative|performance|security|accessibility|regression|uat",
  "priority": "P0 - Critical|P1 - High|P2 - Medium|P3 - Low",
  "preconditions": ["list", "of", "preconditions"],
  "steps": [
    {"step_number": 1, "action": "what the tester does", "expected": "what should happen"}
  ],
  "expected_result": "Overall pass condition",
  "linked_requirement": "Section or requirement ID from the BRD",
  "tags": ["tag1", "tag2"]
}

Return ONLY a JSON array of test case objects. No prose, no markdown fences, no explanations."""


# ── Few-shot examples injected per test type ──────────────────────────────────

FEW_SHOT = {
    TestType.FUNCTIONAL: """Example:
{"title":"Successful user login","test_type":"functional","priority":"P0 - Critical",
"preconditions":["User account exists","Application is running"],
"steps":[{"step_number":1,"action":"Navigate to login page","expected":"Login form visible"},
         {"step_number":2,"action":"Enter valid email and password","expected":"Fields accept input"},
         {"step_number":3,"action":"Click Login","expected":"User redirected to dashboard"}],
"expected_result":"User is authenticated and lands on the dashboard",
"linked_requirement":"REQ-001","tags":["login","auth"]}""",

    TestType.NEGATIVE: """Example:
{"title":"Login with invalid password","test_type":"negative","priority":"P1 - High",
"preconditions":["User account exists"],
"steps":[{"step_number":1,"action":"Enter valid email and wrong password","expected":"Fields accept input"},
         {"step_number":2,"action":"Click Login","expected":"Error message shown"}],
"expected_result":"System shows 'Invalid credentials' and does not authenticate",
"linked_requirement":"REQ-001","tags":["login","negative","security"]}""",

    TestType.EDGE: """Example:
{"title":"Login with email at max length boundary","test_type":"edge","priority":"P2 - Medium",
"preconditions":["System enforces 254-char email limit"],
"steps":[{"step_number":1,"action":"Enter 254-character valid email","expected":"Field accepts input"},
         {"step_number":2,"action":"Enter valid password and click Login","expected":"Authentication proceeds"}],
"expected_result":"System handles max-length email without truncation or error",
"linked_requirement":"REQ-001","tags":["boundary","email"]}""",
}


# ── Main generation function ──────────────────────────────────────────────────

def generate_test_cases(
    brd_chunks: list[str],
    test_types: list[TestType],
    max_cases: int = 50,
    model: str = "claude-sonnet-4-20250514",
) -> tuple[list[TestCase], list[str]]:
    """
    Call the LLM with BRD context and return (test_cases, coverage_gaps).
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable not set.")

    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

    # Build the user prompt
    brd_context = "\n\n---\n\n".join(brd_chunks[:20])  # Limit context window
    type_list   = ", ".join(t.value for t in test_types)
    examples    = "\n\n".join(FEW_SHOT.get(t, "") for t in test_types if t in FEW_SHOT)

    user_prompt = f"""
BRD CONTENT:
{brd_context}

INSTRUCTIONS:
- Generate up to {max_cases} test cases covering these types: {type_list}
- Use chain-of-thought reasoning: first identify all actors, flows, and rules, then generate cases.
- Assign P0 to authentication, payment, data-loss, and security-critical flows.
- Every test case must have a linked_requirement pointing to a section of the BRD above.
- After the JSON array, append a second JSON object on its own line:
  {{"coverage_gaps": ["gap description 1", "gap description 2"]}}
  listing any requirements you found but could not fully cover.

EXAMPLES:
{examples}

Now generate the test cases as a JSON array:
"""

    message = client.messages.create(
        model=model,
        max_tokens=8192,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    raw = message.content[0].text.strip()
    return _parse_llm_output(raw)


# ── Output parser ─────────────────────────────────────────────────────────────

def _parse_llm_output(raw: str) -> tuple[list[TestCase], list[str]]:
    """Parse the LLM's raw text into TestCase objects."""
    # Strip accidental markdown fences
    raw = re.sub(r"```[a-z]*", "", raw).strip()

    # Split off the coverage_gaps trailer if present
    coverage_gaps: list[str] = []
    lines = raw.strip().splitlines()

    # Look for trailing JSON object with coverage_gaps
    trailer_start = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('{"coverage_gaps"') or stripped.startswith('{ "coverage_gaps"'):
            trailer_start = i
            break

    if trailer_start is not None:
        trailer_json = " ".join(lines[trailer_start:])
        main_json    = " ".join(lines[:trailer_start])
        try:
            coverage_gaps = json.loads(trailer_json).get("coverage_gaps", [])
        except json.JSONDecodeError:
            pass
    else:
        main_json = raw

    # Parse test cases array
    try:
        data = json.loads(main_json)
    except json.JSONDecodeError:
        # Attempt to extract first JSON array from the text
        match = re.search(r"\[.*\]", main_json, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            return [], ["LLM returned unparseable output"]

    test_cases: list[TestCase] = []
    for item in data:
        try:
            steps = [TestStep(**s) for s in item.get("steps", [])]
            tc = TestCase(
                title=item.get("title", "Untitled"),
                test_type=TestType(item.get("test_type", "functional")),
                priority=Priority(item.get("priority", "P2 - Medium")),
                preconditions=item.get("preconditions", []),
                steps=steps,
                expected_result=item.get("expected_result", ""),
                linked_requirement=item.get("linked_requirement", ""),
                tags=item.get("tags", []),
            )
            test_cases.append(tc)
        except Exception:
            continue  # Skip malformed cases

    return test_cases, coverage_gaps

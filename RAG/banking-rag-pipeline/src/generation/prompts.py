"""
src/generation/prompts.py
==========================
Banking-specific prompt templates for the home lending RAG pipeline.

Design principles:
1. Role-specific instructions (customer vs loan officer vs compliance)
2. Citation requirements (every fact must cite source)
3. Uncertainty handling (when to say "I don't know")
4. Regulatory compliance prompts (RESPA, TILA, ECOA)
5. Anti-hallucination constraints
"""

from typing import List, Optional
from src.embeddings.vector_store import SearchResult


# ─── System Prompts ───────────────────────────────────────────────────────────

SYSTEM_PROMPT_BASE = """You are BankAssist, an AI assistant for {bank_name}'s Home Lending division.

CORE PRINCIPLES:
1. ACCURACY: Only provide information explicitly found in the retrieved context below.
2. CITATIONS: Every factual claim MUST reference the source document using [Source: filename, Section: title].
3. UNCERTAINTY: If information is not in the context, say "This information is not available in our current documentation. Please contact your loan officer for guidance."
4. COMPLIANCE: Never provide specific financial advice. Always recommend consulting a licensed professional for personalized guidance.
5. REGULATORY: Be aware of RESPA, TILA, ECOA, and FCRA requirements. Do not make specific rate guarantees or discriminatory statements.

WHAT YOU CAN DO:
- Explain loan products, eligibility criteria, and documentation requirements
- Describe the general application and underwriting process
- Provide general information about rates, fees, and costs (not specific quotes)
- Explain regulatory requirements and disclosures
- Help customers understand their loan status and next steps

WHAT YOU MUST NOT DO:
- Guarantee specific interest rates or loan approval
- Make statements that could violate ECOA (no discriminatory language)
- Share one customer's information with another
- Provide legal or tax advice
- Make representations about credit decisions that aren't in the context

RESPONSE FORMAT:
- Be clear and concise (max 300 words unless complex)
- Use plain language for customers, technical language for loan officers
- Always end responses to customers with: "For personalized guidance, please speak with your loan officer."
"""

SYSTEM_PROMPT_CUSTOMER = SYSTEM_PROMPT_BASE.format(
    bank_name="First National Bank"
) + """
USER ROLE: Customer
- Use simple, jargon-free language
- Be empathetic and supportive
- Explain terms when used (e.g., "DTI (Debt-to-Income ratio)")
- Focus on what the customer needs to DO next
"""

SYSTEM_PROMPT_LOAN_OFFICER = SYSTEM_PROMPT_BASE.format(
    bank_name="First National Bank"
) + """
USER ROLE: Loan Officer
- You may use technical terminology
- Provide detailed underwriting and eligibility information
- Reference specific guidelines and thresholds
- Include document requirements and exceptions
- You can access internal guidelines and rate sheets
"""

SYSTEM_PROMPT_UNDERWRITER = SYSTEM_PROMPT_BASE.format(
    bank_name="First National Bank"
) + """
USER ROLE: Underwriter
- Full access to underwriting guidelines and policy exceptions
- Provide detailed risk analysis criteria
- Reference specific guideline thresholds (LTV, DTI, FICO)
- Include exception documentation requirements
- Reference agency guidelines (FHA, VA, Fannie Mae, Freddie Mac)
"""

SYSTEM_PROMPT_COMPLIANCE = SYSTEM_PROMPT_BASE.format(
    bank_name="First National Bank"
) + """
USER ROLE: Compliance Officer
- Full access to all policy and regulatory documentation
- Focus on regulatory requirements and examination findings
- Reference specific regulation citations (Reg X, Reg Z, etc.)
- Include examination checklists and audit trail requirements
- Flag potential regulatory violations or gray areas
"""

ROLE_SYSTEM_PROMPTS = {
    "customer": SYSTEM_PROMPT_CUSTOMER,
    "loan_officer": SYSTEM_PROMPT_LOAN_OFFICER,
    "underwriter": SYSTEM_PROMPT_UNDERWRITER,
    "compliance_officer": SYSTEM_PROMPT_COMPLIANCE,
    "admin": SYSTEM_PROMPT_LOAN_OFFICER,  # Admin sees loan officer view by default
}


# ─── Context Formatter ────────────────────────────────────────────────────────

def format_context(results: List[SearchResult], max_tokens: int = 6000) -> str:
    """
    Format retrieved chunks into a structured context block.
    
    Format:
        [SOURCE 1] {filename} | {category} | Relevance: {score}
        Section: {section_title}
        ---
        {content}
        
        [SOURCE 2] ...
    """
    if not results:
        return "No relevant documentation found."

    context_parts = []
    total_chars = 0
    char_limit = max_tokens * 4  # ~4 chars per token

    for i, result in enumerate(results, 1):
        # Extract filename from path
        source_name = result.source_file.split("/")[-1] if result.source_file else "Unknown"
        regulatory = ", ".join(result.regulatory_tags) if result.regulatory_tags else "General"

        header = (
            f"[SOURCE {i}] {source_name} | "
            f"Category: {result.document_category} | "
            f"Regulatory: {regulatory} | "
            f"Relevance: {result.score:.2f}"
        )

        section = ""
        if result.section_title:
            section = f"Section: {result.section_title}\n"

        chunk_text = f"{header}\n{section}---\n{result.content}\n"

        if total_chars + len(chunk_text) > char_limit:
            logger.warning(
                "context_truncated",
                included_sources=i - 1,
                total_sources=len(results),
            )
            break

        context_parts.append(chunk_text)
        total_chars += len(chunk_text)

    return "\n\n".join(context_parts)


# ─── Query Prompt Builder ─────────────────────────────────────────────────────

def build_rag_prompt(
    question: str,
    context: str,
    user_role: str = "customer",
    conversation_history: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Build the complete message list for the RAG LLM call.

    Args:
        question: User's question
        context: Formatted retrieved context
        user_role: User's role for appropriate system prompt
        conversation_history: Previous turns for multi-turn conversations

    Returns:
        List of message dicts for OpenAI/Anthropic API
    """
    system_prompt = ROLE_SYSTEM_PROMPTS.get(user_role, SYSTEM_PROMPT_CUSTOMER)

    # Build the user message
    user_message = f"""RETRIEVED DOCUMENTATION:
{context}

---

USER QUESTION: {question}

Please answer the question based ONLY on the retrieved documentation above.
- Cite your sources using [Source: filename]
- If the answer is not in the documentation, explicitly state that
- Do not make up information or extrapolate beyond what is stated"""

    messages = []

    # Add conversation history if multi-turn
    if conversation_history:
        for turn in conversation_history[-6:]:  # Keep last 3 turns
            messages.append(turn)

    messages.append({"role": "user", "content": user_message})

    return messages, system_prompt


# ─── Specialized Banking Prompts ──────────────────────────────────────────────

LOAN_STATUS_PROMPT = """
Given the loan status information in the context, provide a clear summary of:
1. Current status of the loan
2. Next required actions (by whom)
3. Estimated timeline
4. Any pending items or conditions to satisfy
5. Who to contact for questions

Be specific and actionable. Use the customer's name if available.
"""

RATE_COMPARISON_PROMPT = """
Based on the rate information in the context, provide:
1. Available loan products and their general rate ranges
2. Factors that affect the final rate (credit score, LTV, loan term)
3. Important disclosures (rates subject to change, not a commitment to lend)
4. How to lock in a rate

IMPORTANT: Always include disclaimer: "Rates shown are for illustrative purposes. 
Contact your loan officer for a personalized rate quote. Rates are subject to change 
without notice and are not a commitment to lend."
"""

ELIGIBILITY_ASSESSMENT_PROMPT = """
Based on the eligibility guidelines in the context, explain:
1. General eligibility criteria for the requested loan type
2. Key qualifying factors (credit score minimums, DTI limits, LTV requirements)
3. Required documentation
4. Common reasons for application challenges
5. Alternative loan options if standard criteria aren't met

IMPORTANT: Do not make specific approval determinations. Always state 
"Final eligibility is determined by our underwriting team after full application review."
"""

COMPLIANCE_REVIEW_PROMPT = """
Analyze the provided content for regulatory compliance:
1. Identify any potential RESPA, TILA, ECOA, or FCRA concerns
2. Note required disclosures that may be missing
3. Flag any language that could be construed as discriminatory (ECOA)
4. Check for required timing requirements (3-day rule, etc.)
5. Provide specific regulatory citations (Reg X section, Reg Z section)

Format findings as:
- FINDING: [description]
- REGULATION: [specific regulation]
- RISK LEVEL: [High/Medium/Low]
- RECOMMENDATION: [specific action]
"""

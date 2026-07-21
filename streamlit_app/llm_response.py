"""
llm_response.py
----------------
Drafts a suggested agent reply for a ticket, using the outputs of the rest
of the pipeline (issue type, emotion, priority, escalation risk, resolution
estimate, similar tickets) as grounding context.

Two modes:
  1. Live LLM call (Anthropic Claude or OpenAI) if an API key is provided.
  2. Rule-based template fallback if no key is provided, so the app is
     always usable in a demo/offline setting.
"""

SYSTEM_PROMPT = (
    "You are an assistant that drafts helpful, empathetic first-response "
    "messages for a customer support agent. You are writing FOR the agent "
    "to review and send, not directly to the customer. Keep the tone "
    "professional, concise (under 150 words), and acknowledge the "
    "customer's emotional state without being over the top. Do not invent "
    "specific account details, refund amounts, or policy promises you "
    "aren't given. End with a clear next step."
)


def _build_user_prompt(ticket_text, context: dict):
    similar_lines = "\n".join(
        f"- ({t['label']}, {t['similarity']:.0f}% similar): {t['text']}"
        for t in context.get("similar_tickets", [])
    ) or "- None found above the similarity threshold"

    prompt = f"""Customer ticket:
"{ticket_text}"

Predicted issue category: {context['top_issue_type']}
Detected customer emotion: {context['emotion_predictions'][0]['label']} ({context['emotion_predictions'][0]['confidence']}% confidence)
Priority: {context['tabular_row']['priority']}
Escalation risk: {context['escalation']['label']}
Estimated resolution time: {context['resolution']['label']} ({context['resolution']['hours']})

Similar past tickets:
{similar_lines}

Write a first-response reply the agent can send to this customer."""
    return prompt


def generate_template_response(ticket_text, context: dict):
    """No API key needed. Rule-based, but grounded in the model outputs."""
    issue_type = context["top_issue_type"]
    emotion = context["emotion_predictions"][0]["label"]
    priority = context["tabular_row"]["priority"]
    resolution_hours = context["resolution"]["hours"]
    escalation_label = context["escalation"]["label"]

    openers = {
        "anger": "I completely understand your frustration, and I'm sorry for the trouble this has caused.",
        "disgust": "I'm sorry this experience has been so frustrating — that's not the experience we want you to have.",
        "fear": "I understand this is concerning, and I want to help get this sorted out for you right away.",
        "sadness": "I'm really sorry to hear you're dealing with this.",
        "surprise": "Thanks for flagging this — I can see why that would catch you off guard.",
        "joy": "Thanks so much for reaching out!",
        "neutral": "Thanks for reaching out to us.",
    }

    issue_body = {
        "account_access": "It looks like this is related to account access. Our team will verify your account details and help restore access as quickly as possible.",
        "billing_problem": "It looks like this is a billing-related issue. We'll review the charges on your account and follow up with a correction or explanation.",
        "bug": "This looks like a product bug. Our engineering team will investigate the behavior you described and work on a fix.",
        "feature_request": "Thanks for the suggestion — I've logged this as a feature request for our product team to review.",
        "how_to": "I'll walk you through this, or point you to the right guide so you can get this done quickly.",
        "performance": "It looks like you're experiencing a performance issue. We'll look into what might be causing the slowdown.",
        "security_concern": "Security concerns are treated as a top priority. Our security team will look into this immediately.",
        "other": "Thanks for the details — I'll make sure this gets to the right team to take a closer look.",
    }

    opener = openers.get(emotion, openers["neutral"])
    body = issue_body.get(issue_type, issue_body["other"])

    urgency_note = ""
    if escalation_label == "High":
        urgency_note = " Given the nature of this issue, I'm escalating it internally so it gets immediate attention."
    elif priority in ("high", "urgent"):
        urgency_note = " I've flagged this ticket as high priority."

    closer = f" Based on similar cases, we typically resolve this kind of issue within {resolution_hours}. I'll keep you updated on progress."

    return f"{opener} {body}{urgency_note}{closer}".strip()


def generate_llm_response(ticket_text, context: dict, provider: str, api_key: str, model_name: str = None):
    """
    provider: "anthropic" or "openai"
    Raises on failure so the caller can fall back to the template response.
    """
    user_prompt = _build_user_prompt(ticket_text, context)

    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model_name or "claude-sonnet-4-6",
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text_parts = [block.text for block in response.content if block.type == "text"]
        return "\n".join(text_parts).strip()

    elif provider == "openai":
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model_name or "gpt-4o-mini",
            max_tokens=400,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    else:
        raise ValueError(f"Unknown provider: {provider}")


def get_suggested_reply(ticket_text, context: dict, provider: str, api_key: str, model_name: str = None):
    """
    Top-level entry point used by the Streamlit app.
    Returns (reply_text, source) where source is "llm" or "template".
    """
    if provider in ("anthropic", "openai") and api_key:
        try:
            reply = generate_llm_response(ticket_text, context, provider, api_key, model_name)
            return reply, "llm"
        except Exception as e:
            fallback = generate_template_response(ticket_text, context)
            return fallback, f"template (LLM call failed: {e})"

    return generate_template_response(ticket_text, context), "template"
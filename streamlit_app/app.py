import streamlit as st
from utils import load_all_models, get_categorical_options, run_pipeline,suggest_priority, EMOTION_EMOJI
from llm_response import get_suggested_reply
 
st.set_page_config(
    page_title="AI Customer Support Intelligence Platform",
    page_icon="🤖",
    layout="wide",
)
 
# ---------------------------------------------------------------------------
# Sidebar: LLM configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Suggested Reply Settings")
    provider_choice = st.selectbox(
        "Reply generation mode",
        ["Template only (no API key needed)", "Anthropic Claude", "OpenAI"],
    )
    api_key = ""
    model_name = None
    provider = "template"
    if provider_choice == "Anthropic Claude":
        provider = "anthropic"
        api_key = st.text_input("Anthropic API key", type="password")
        model_name = st.text_input("Model", value="claude-sonnet-4-6")
    elif provider_choice == "OpenAI":
        provider = "openai"
        api_key = st.text_input("OpenAI API key", type="password")
        model_name = st.text_input("Model", value="gpt-4o-mini")
 
    st.caption("API keys are used only for this session's request and are never stored.")
 
    st.divider()
    st.header("ℹ️ About")
    st.caption(
        "Internal tool for support agents. Classifies incoming tickets, "
        "estimates escalation risk & resolution time, surfaces similar past "
        "tickets, and drafts a first-response reply."
    )
 
# ---------------------------------------------------------------------------
# Warm up all cached models once
# ---------------------------------------------------------------------------
with st.spinner("Warming up models..."):
    load_all_models()
    categorical_options = get_categorical_options()
 
st.title("🤖 AI Customer Support Intelligence Platform")
st.caption("Company-facing ticket triage dashboard — not visible to customers.")
 
# ---------------------------------------------------------------------------
# Ticket intake form
# ---------------------------------------------------------------------------
st.subheader("📥 Incoming Ticket")

ticket_text = st.text_area(
    "Ticket message (from email / chat / call transcript / help desk form)",
    height=120,
    placeholder="e.g. I was charged twice for my subscription this month and I need a refund.",
    key="ticket_text_input",
)

col1, col2, col3, col4 = st.columns(4)
with col1:
    channel = st.selectbox("Channel", categorical_options["channel"])
    customer_segment = st.selectbox("Customer segment", categorical_options["customer_segment"])
with col2:
    platform = st.selectbox("Platform", categorical_options["platform"])
    region = st.selectbox("Region", categorical_options["region"])
with col3:
    product_area = st.selectbox("Product area", categorical_options["product_area"])
    sla_plan = st.selectbox("SLA plan", categorical_options["sla_plan"])
with col4:
    has_attachment_label = st.selectbox("Attachment?", categorical_options["has_attachment"])

has_attachment = 1 if has_attachment_label == "Yes" else 0

# --- Rule-based priority suggestion (not ML — see pipeline.suggest_priority) --
suggested = suggest_priority(ticket_text, sla_plan, customer_segment, product_area) if ticket_text.strip() else None

if suggested:
    st.caption(f"💡 Suggested priority (rule-based on keywords + SLA + segment): **{suggested.title()}** — confirm or change below.")
else:
    st.caption("💡 Enter ticket text above to see a suggested priority.")

priority_options = categorical_options["priority"]
default_index = priority_options.index(suggested) if suggested in priority_options else 0
priority = st.selectbox("Priority", priority_options, index=default_index)

submitted = st.button("Analyze Ticket", type="primary", use_container_width=True)
 
# ---------------------------------------------------------------------------
# Run pipeline + render results
# ---------------------------------------------------------------------------
if submitted:
    if not ticket_text.strip():
        st.error("Please enter a ticket message before analyzing.")
        st.stop()
 
    form_inputs = {
        "priority": priority,
        "channel": channel,
        "customer_segment": customer_segment,
        "platform": platform,
        "region": region,
        "product_area": product_area,
        "has_attachment": has_attachment,
        "sla_plan": sla_plan,
        "reopened": 0,  # new ticket, hasn't been reopened yet
    }
 
    with st.spinner("Analyzing ticket..."):
        result = run_pipeline(ticket_text, form_inputs)
 
    st.success("Analysis complete.")
    st.divider()
 
    # --- Top row: key signals ------------------------------------------------
    m1, m2, m3, m4 = st.columns(4)
 
    with m1:
        st.markdown("**🏷️ Issue Category**")
        st.markdown(f"### {result['top_issue_type'].replace('_', ' ').title()}")
        st.caption(f"{result['issue_predictions'][0]['confidence']}% confidence")
 
    with m2:
        top_emotion = result["emotion_predictions"][0]
        emoji = EMOTION_EMOJI.get(top_emotion["label"], "")
        st.markdown("**🎭 Customer Emotion**")
        st.markdown(f"### {emoji} {top_emotion['label'].title()}")
        st.caption(f"{top_emotion['confidence']}% confidence · sentiment: {result['derived_sentiment']}")
 
    with m3:
        esc = result["escalation"]
        st.markdown("**🚨 Escalation Risk**")
        st.markdown(f"### {esc['color']} {esc['label']}")
        st.caption(esc["description"])
 
    with m4:
        res = result["resolution"]
        st.markdown("**⏱️ Est. Resolution Time**")
        st.markdown(f"### {res['color']} {res['label']}")
        st.caption(res["hours"])
 
    st.divider()
 
    # --- Detail row: expandable breakdowns ------------------------------------
    d1, d2 = st.columns(2)
 
    with d1:
        with st.expander("📊 Issue category — top 3", expanded=True):
            for p in result["issue_predictions"]:
                st.write(p["label"].replace("_", " ").title())
                st.progress(min(int(p["confidence"]), 100), text=f"{p['confidence']}%")
 
        with st.expander("📊 Escalation risk — probability breakdown"):
            for label, pct in result["escalation"]["probabilities"].items():
                st.write(label)
                st.progress(min(int(pct), 100), text=f"{pct}%")
 
    with d2:
        with st.expander("📊 Customer emotion — top 3", expanded=True):
            for p in result["emotion_predictions"]:
                emoji = EMOTION_EMOJI.get(p["label"], "")
                st.write(f"{emoji} {p['label'].title()}")
                st.progress(min(int(p["confidence"]), 100), text=f"{p['confidence']}%")
 
        with st.expander("📊 Resolution time — probability breakdown"):
            for label, pct in result["resolution"]["probabilities"].items():
                st.write(label)
                st.progress(min(int(pct), 100), text=f"{pct}%")
 
    st.divider()
 
    # --- Similar tickets --------------------------------------------------------
    st.subheader("🔍 Similar Past Tickets")
    similar = result["similar_tickets"]
    if not similar:
        st.info("No sufficiently similar past tickets were found.")
    else:
        for t in similar:
            with st.container(border=True):
                c1, c2 = st.columns([5, 1])
                with c1:
                    st.write(t["text"])
                    st.caption(f"Category: {t['label'].replace('_', ' ').title()}")
                with c2:
                    st.metric("Similarity", f"{t['similarity']:.0f}%")
 
    st.divider()
 
    # --- Suggested reply --------------------------------------------------------
    st.subheader("✍️ Suggested First-Response Reply")
    with st.spinner("Drafting reply..."):
        reply_text, source = get_suggested_reply(ticket_text, result, provider, api_key, model_name)
 
    if source == "llm":
        st.caption(f"Generated by {provider_choice}")
    else:
        st.caption(f"Generated from template ({source})" if "failed" in source else "Generated from rule-based template (no API key provided)")
 
    st.text_area("Editable reply", value=reply_text, height=180, key="reply_box")
    st.download_button("📋 Download reply as .txt", data=reply_text, file_name="suggested_reply.txt", mime="text/plain")
 
else:
    st.info("Enter a ticket above and click **Analyze Ticket** to run the full pipeline.")
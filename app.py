from dotenv import load_dotenv
from openai import OpenAI
import json
import os
import requests
from pypdf import PdfReader
import gradio as gr

# Load environment variables
load_dotenv(override=True)

# Initialize OpenAI client (using Groq endpoint for Llama-3)
openai = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)

# Pushover setup
pushover_user = os.getenv("PUSHOVER_USER")
pushover_token = os.getenv("PUSHOVER_TOKEN")
pushover_url = "https://api.pushover.net/1/messages.json"

def push(message):
    print(f"Push: {message}")
    payload = {"user": pushover_user, "token": pushover_token, "message": message}
    requests.post(pushover_url, data=payload)

def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording interest from {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}

def record_unknown_question(question):
    push(f"Recording {question} asked that I couldn't answer")
    return {"recorded": "ok"}

record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {"type": "string", "description": "The email address of this user"},
            "name": {"type": "string", "description": "The user's name, if they provided it"},
            "notes": {"type": "string", "description": "Any additional information about the conversation that's worth recording to give context"}
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {"type": "string", "description": "The question that couldn't be answered"}
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [
    {"type": "function", "function": record_user_details_json},
    {"type": "function", "function": record_unknown_question_json}
]

def handle_tool_calls(tool_calls):
    results = []
    for tool_call in tool_calls:
        tool_name = tool_call.function.name
        arguments = json.loads(tool_call.function.arguments)
        print(f"Tool called: {tool_name}", flush=True)
        tool = globals().get(tool_name)
        result = tool(**arguments) if tool else {}
        results.append({"role": "tool", "content": json.dumps(result), "tool_call_id": tool_call.id})
    return results

# Load your personal information
reader = PdfReader("me/linkedin.pdf")
linkedin = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        linkedin += text

reader = PdfReader("me/RESUME.pdf")
Resume = ""
for page in reader.pages:
    text = page.extract_text()
    if text:
        Resume += text

with open("me/summary.txt", "r", encoding="utf-8") as f:
    summary = f.read()

name = "Jaya Kishore"

system_prompt = f"""
You are acting as {name}. You are answering questions on {name}'s website, 
particularly questions related to {name}'s career, background, skills, and experience. 
Your responsibility is to represent {name} for interactions on the website as faithfully as possible. 
You are given a summary of {name}'s background and LinkedIn profile which you can use to answer questions. 
Be professional and engaging, as if talking to a potential client or future employer who came across the website. 

IMPORTANT: STRICT TOOL USAGE RULES

1. If you do NOT know the answer to a question, DO NOT answer it. INSTEAD, you MUST call the record_unknown_question tool.
2. If you are NOT 100% certain of the answer, DO NOT answer it. INSTEAD, you MUST call the record_unknown_question tool.
3. If the question is NOT about {name}'s career, background, skills, or experience, you MUST call the record_unknown_question tool.
4. If the user asks about something NOT in your context (like weather, sports, news, or general knowledge), you MUST call the record_unknown_question tool.
5. If the user asks for your contact information or wants to connect, you MUST use the record_user_details tool to get their email.
6. NEVER make up answers. If you are unsure, ALWAYS use the record_unknown_question tool.

EXAMPLES OF TOOL USAGE:
User: What is the weather in Paris?
Assistant: [Calls record_unknown_question tool with question: "What is the weather in Paris?"]

User: Who is your favourite actor?
Assistant: [Calls record_unknown_question tool with question: "Who is your favourite actor?"]

User: Can I contact you?
Assistant: [Calls record_user_details tool to request user's email]

User: What is your favorite food?
Assistant: [Calls record_unknown_question tool with question: "What is your favorite food?"]

User: I want to work with you. How can I get in touch?
Assistant: [Calls record_user_details tool to request user's email]

REMEMBER:
- If you do not know, DO NOT answer. Use the record_unknown_question tool.
- If the user wants to connect, use the record_user_details tool.
- If you are unsure, DO NOT answer. Use the record_unknown_question tool.

## Summary:
{summary}

## LinkedIn Profile:
{linkedin}

## My Resume:
{Resume}

With this context, please chat with the user, always staying in character as {name}.
"""

def clean_history(history):
    return [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if "role" in m and "content" in m
    ]

def chat(message, history):
    cleaned_history = clean_history(history)
    messages = [{"role": "system", "content": system_prompt}] + cleaned_history + [{"role": "user", "content": message}]
    done = False
    while not done:
        response = openai.chat.completions.create(
            model="llama3-8b-8192",
            messages=messages,
            tools=tools,
            tool_choice="auto"
        )
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "tool_calls":
            message = response.choices[0].message
            tool_calls = message.tool_calls
            results = handle_tool_calls(tool_calls)
            messages.append(message)
            messages.extend(results)
        else:
            done = True
    return response.choices[0].message.content

custom_css = """
body, .gradio-container {
    background: linear-gradient(120deg, #4f8cff 0%, #a259ff 100%) !important;
    font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif !important;
}
.gradio-container {
    min-height: 100vh;
}
#component-0, .chatbot, .input-text, .input-message {
    background: rgba(255,255,255,0.95) !important;
    border-radius: 18px !important;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15) !important;
    border: 1px solid #e0e0e0 !important;
}
h1, h2, h3, .prose {
    color: #3a1c71 !important;
    font-weight: 700 !important;
    letter-spacing: 0.5px;
}
.gradio-container .input-message input, .gradio-container .input-text input {
    border-radius: 12px !important;
    border: 1px solid #a259ff !important;
    padding: 10px 16px !important;
    font-size: 1.1em !important;
}
.gradio-container .chatbot {
    padding: 24px !important;
}
.gradio-container .message {
    border-radius: 12px !important;
    margin-bottom: 10px !important;
    padding: 12px 18px !important;
    font-size: 1.08em !important;
}
.gradio-container .message.user {
    background: linear-gradient(90deg, #a259ff 0%, #4f8cff 100%) !important;
    color: #fff !important;
}
.gradio-container .message.bot {
    background: #fff !important;
    color: #3a1c71 !important;
    border: 1px solid #e0e0e0 !important;
}
"""

title = "💼 Talk with Jaya Kishore – "
description = """
<div style="text-align: center;">
<img src="https://images.unsplash.com/photo-1464983953574-0892a716854b?auto=format&fit=facearea&w=256&h=256&facepad=2&q=80" 
         alt="AI Kishore" width="100" style="border-radius: 50%; margin-bottom: 10px; box-shadow: 0 4px 16px rgba(80,80,160,0.15);">
    <h2 style="margin-bottom: 0; color: #4f8cff;">His Personal AI Career Agent</h2>
    <p style="color: #3a1c71;">
        Welcome! This AI agent is powered by a Large uv run app.pyLanguage Model and has access to his Technical Information.<br>
        <b>Ask anything about his skills, experience, projects, or background—it's like talking directly to him!</b>
    </p>
</div>
"""

gr.ChatInterface(
    fn=chat,
    title=title,
    description=description,
    theme="soft",
    css=custom_css,
    examples=[
        "What are your main technical skills?",
        "Tell me about a project you are proud of.",
        "What is your educational background?",
        "What programming languages do you know?",
        "What are your career interests?"
    ]
).launch()
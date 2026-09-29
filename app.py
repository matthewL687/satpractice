import os
import requests
from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__, static_folder="static")

OPENAI_API_ENDPOINT = "https://api.openai.com/v1/responses"


# =========================
# CORS
# =========================
@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin")
    resp.headers["Access-Control-Allow-Origin"] = origin or "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return resp


@app.route("/api/<path:_>", methods=["OPTIONS"])
def api_preflight(_):
    return ("", 204)


# =========================
# Core logic
# =========================
def _extract_response_text(data: dict) -> str:
    """
    Robustly extract text from the Responses API payload.
    Works across different response shapes.
    """
    if isinstance(data, dict) and isinstance(data.get("output_text"), str) and data["output_text"].strip():
        return data["output_text"]

    out = data.get("output")
    if isinstance(out, list):
        chunks = []
        for item in out:
            content = item.get("content") if isinstance(item, dict) else None
            if not isinstance(content, list):
                continue
            for c in content:
                if not isinstance(c, dict):
                    continue
                if isinstance(c.get("text"), str):
                    chunks.append(c["text"])
        text = "\n".join(chunks).strip()
        if text:
            return text

    raise RuntimeError("Could not extract text from model response.")


def generate_sat_question(section: str, topic: str, difficulty: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    # Dynamic rules based on the test section
    section_rules = ""
    if "math" in section.lower():
        section_rules = """
- Include realistic contexts (word problems) or clean algebraic expressions typical of the digital SAT.
- Ensure all math notation uses LaTeX ($...$ for inline, $$...$$ for standalone).
- Make sure incorrect answer choices (distractors) stem from common student errors (e.g., sign errors, forgetting to multiply, misapplying formulas).
"""
    else:
        section_rules = """
- Include a short, self-contained, college-level reading passage (25–75 words) appropriate for the digital SAT format.
- Ensure the question directly targets the passage based on standard Digital SAT task types (e.g., Words in Context, Command of Evidence, Central Ideas, Inferences, Expression of Ideas).
- Distractors must be plausible, subtle, and based on common reading misinterpretations rather than obviously wrong choices.
"""

    prompt = f"""
You are an expert College Board SAT item writer specializing in the Digital SAT format. 
Generate ONE high-quality, authentic SAT question based on the specification below.

Section: {section}
Topic: {topic}
Difficulty: {difficulty}

CONTENT REQUIREMENTS:
{section_rules}
- Difficulty scaling:
  * Easy: Straightforward application of standard formulas or direct passage evidence.
  * Medium: Multi-step reasoning or moderate context complexity.
  * Hard: Complex multi-step reasoning, subtle distractors, or higher-level abstractions.

STRICT FORMATTING REQUIREMENTS:
- Use standard SAT tone and formatting.
- Exactly 4 answer choices labeled A., B., C., D.
- Do NOT use HTML tags.
- Do NOT include markdown code blocks or extra metadata text.

Output format (plain text strictly following this template):

Question:
<Passage, prompt, or math stem here>

Answer Choices:
A. <Option A>
B. <Option B>
C. <Option C>
D. <Option D>

Correct Answer:
<Single letter: A, B, C, or D>

Explanation:
<Step-by-step breakdown explaining why the correct answer is right and why key distractors are incorrect>
""".strip()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": "gpt-4.1-mini",
        "input": prompt,
        "max_output_tokens": 800,  # Increased token limit to allow for passages and detailed explanations
        "temperature": 0.5,        # Slightly raised temperature for better variety in question contexts
    }

    r = requests.post(
        OPENAI_API_ENDPOINT,
        headers=headers,
        json=payload,
        timeout=60,
    )

    if r.status_code >= 400:
        raise RuntimeError(r.text)

    data = r.json()
    return _extract_response_text(data)


# =========================
# API route
# =========================
@app.post("/api/generate-question")
def api_generate_question():
    body = request.get_json() or {}

    section = (body.get("section") or "").strip()
    topic = (body.get("topic") or "").strip()
    difficulty = (body.get("difficulty") or "").strip()

    if not section or not topic or not difficulty:
        return jsonify({"error": "Missing fields"}), 400

    try:
        text = generate_sat_question(section, topic, difficulty)
        return jsonify({"text": text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Frontend / static
# =========================
@app.get("/")
def index():
    return send_from_directory("static", "index.html")


@app.get("/<path:path>")
def static_proxy(path):
    return send_from_directory("static", path)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

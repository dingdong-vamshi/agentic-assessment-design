from google import genai

def recomend_agent(state: dict) -> dict:
    problems = state.get("problems", [])
    principles = state.get("principles", [])
    prompt = f"""
        You are an expert in educational assessment design.
        Problems identified in the exam:
        {chr(10).join(f"- {p}" for p in problems)}
        Relevant teaching principles:
        {chr(10).join(f"- {r}" for r in principles)}
        Task:
        Give exactly 3 clear, actionable recommendations to improve the exam.
        Rules:
        - Be concise
        - Output as a numbered list"""
    client = genai.Client()
    response = client.models.generate_content(
        model="gemini-3-flash-preview", contents=prompt
    )
    result = response.text.strip().split("\n")
    recommendations = [line.strip() for line in result if line.strip()]
    state["recommendations"] = recommendations
    return state
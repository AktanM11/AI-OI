import os
import io
import contextlib
import gradio as gr
import requests
import inspect
import pandas as pd

# (Keep Constants as is)
# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Basic Agent Definition ---
# ----- THIS IS WERE YOU CAN BUILD WHAT YOU WANT ------
import os
from typing import TypedDict, Annotated, Any

from duckduckgo_search import DDGS

from langchain.tools import tool
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_openai import ChatOpenAI

# =====================================================
# HF TOKEN FROM SPACE SECRET
# =====================================================

FM_TOKEN = os.environ["FM_TOKEN"]

# =====================================================
# WEB SEARCH TOOL
# =====================================================

@tool
def web_search(query: str) -> str:
    """
    Search the web and return useful snippets.
    """

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return "No results found."

        output = []

        for r in results:
            output.append(
                f"""
Title: {r.get("title","")}
Snippet: {r.get("body","")}
URL: {r.get("href","")}
"""
            )

        return "\n".join(output)

    except Exception as e:
        return f"Search error: {e}"


# =====================================================
# PYTHON TOOL
# =====================================================


@tool
def python_tool(code: str) -> str:
    """
    Execute Python code and return stdout or errors.
    """
    # Run user code in an isolated namespace and capture stdout.
    if not code or not code.strip():
        return "No code provided."

    local_vars: dict[str, Any] = {}
    stdout = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout):
            exec(code, {}, local_vars)
    except Exception as exc:
        return f"Python error: {exc}"

    output = stdout.getvalue().strip()
    if output:
        return output
    return "Execution completed. No output."


# =====================================================
# FETCH URL TOOL
# =====================================================


@tool
def fetch_url(url: str, max_chars: int = 4000) -> str:
    """
    Fetch a URL and return the response text.
    """
    # Limit content length to keep responses manageable.
    if not url or not url.strip():
        return "No URL provided."

    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        return f"Fetch error: {exc}"

    text = response.text or ""
    if not text.strip():
        return "No text content found."
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "\n\n[truncated]"
    return text


# =====================================================
# PDF READER TOOL
# =====================================================


def _load_pdf_bytes(source: str) -> bytes:
    """
    Load PDF bytes from a URL or local path.
    """
    # Support URLs, file:// URIs, and local paths.
    if source.startswith("http://") or source.startswith("https://"):
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        return response.content

    if source.startswith("file://"):
        source = source.replace("file://", "", 1)

    if not os.path.exists(source):
        raise FileNotFoundError(f"File not found: {source}")

    with open(source, "rb") as handle:
        return handle.read()


@tool
def pdf_reader(
    source: str,
    max_pages: int = 5,
    max_chars: int = 4000,
) -> str:
    """
    Extract text from a PDF at a URL or local path.
    """
    # Use pypdf for extraction; return helpful errors on failure.
    if not source or not source.strip():
        return "No PDF source provided."

    try:
        from pypdf import PdfReader
    except ImportError:
        return "Missing dependency: install pypdf."

    try:
        pdf_bytes = _load_pdf_bytes(source)
    except Exception as exc:
        return f"PDF load error: {exc}"

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
    except Exception as exc:
        return f"PDF parse error: {exc}"

    text_parts: list[str] = []
    page_count = min(len(reader.pages), max_pages)
    for i in range(page_count):
        page_text = reader.pages[i].extract_text() or ""
        if page_text:
            text_parts.append(page_text)

    if not text_parts:
        return "No extractable text found."

    text = "\n\n".join(text_parts)
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "\n\n[truncated]"
    return text


# =====================================================
# TASK FILE TOOL
# =====================================================


@tool
def fetch_task_file(task_id: str, max_chars: int = 4000) -> str:
    """
    Fetch a task file by task_id from the scoring API.
    """
    # Use the scoring API to retrieve task-specific files.
    if not task_id or not task_id.strip():
        return "No task_id provided."

    url = f"{DEFAULT_API_URL}/files/{task_id.strip()}"
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        return f"File fetch error: {exc}"

    text = response.text or ""
    if not text.strip():
        return "No text content found."
    if max_chars and len(text) > max_chars:
        return text[:max_chars] + "\n\n[truncated]"
    return text


# =====================================================
# LLM
# =====================================================

llm = ChatOpenAI(
    api_key=FM_TOKEN,
    model="gpt-5.5"
)

chat = llm

tools = [web_search, python_tool, fetch_url, pdf_reader, fetch_task_file]

chat_with_tools = chat.bind_tools(tools)

# =====================================================
# STATE
# =====================================================

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

# =====================================================
# ASSISTANT NODE
# =====================================================

def assistant(state: AgentState):
    """
    Run the assistant with a strict output-only instruction.
    """
    system_msg = SystemMessage(
        content=(
            "You are an expert research and reasoning agent.\n\n"
            "You have access to tools for:\n"
            "- web search\n"
            "- webpage retrieval\n"
            "- PDF reading\n"
            "- Python execution\n"
            "- task file retrieval\n\n"
            "Always use tools when information is missing, uncertain, "
            "requires computation, or depends on external documents.\n\n"
            "When solving a task:\n\n"
            "1. Read the question carefully.\n"
            "2. Use tools whenever necessary.\n"
            "3. Verify facts before answering.\n"
            "4. Perform calculations with Python rather than mental math.\n"
            "5. Read files and PDFs when relevant.\n"
            "6. Continue gathering evidence until you are confident.\n\n"
            "If the task references a file, use the task file tool.\n\n"
            "IMPORTANT OUTPUT RULES:\n\n"
            "- Return ONLY the final answer.\n"
            "- Do NOT explain your reasoning.\n"
            "- Do NOT provide analysis.\n"
            "- Do NOT provide step-by-step solutions.\n"
            "- Do NOT provide preambles.\n"
            "- Do NOT provide markdown.\n"
            "- Do NOT provide bullet points.\n"
            "- Do NOT provide quotes unless the answer itself requires quotes.\n"
            "- Do NOT write 'FINAL ANSWER'.\n"
            "- Do NOT write 'The answer is'.\n"
            "- Do NOT write any surrounding text.\n\n"
            "Your entire response must contain only the exact answer.\n\n"
            "Examples:\n\n"
            "Correct:\n"
            "Paris\n\n"
            "Incorrect:\n"
            "The answer is Paris\n\n"
            "Incorrect:\n"
            "FINAL ANSWER: Paris\n\n"
            "Correct:\n"
            "42\n\n"
            "Incorrect:\n"
            "42.\n\n"
            "Correct:\n"
            "1997\n\n"
            "Incorrect:\n"
            "The year is 1997\n\n"
            "If the answer is a number, output only the number.\n\n"
            "If the answer is a name, output only the name.\n\n"
            "If the answer is a date, output only the date.\n\n"
            "If the answer is a phrase, output only the phrase.\n\n"
            "Be precise."
        )
    )

    response = chat_with_tools.invoke(
        [system_msg, *state["messages"]]
    )

    return {
        "messages": [response]
    }

# =====================================================
# GRAPH
# =====================================================

builder = StateGraph(AgentState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")

builder.add_conditional_edges(
    "assistant",
    tools_condition
)

builder.add_edge(
    "tools",
    "assistant"
)

graph = builder.compile()

# =====================================================
# BASIC AGENT FOR HF COURSE
# =====================================================

class BasicAgent:

    def __init__(self):
        self.graph = graph

    def __call__(self, task_id: str, question: str) -> str:

        try:
            prompt = (
                f"Task ID: {task_id}\n\n"
                "Question:\n"
                f"{question}"
            )
            result = self.graph.invoke(
                {
                    "messages": [
                        HumanMessage(content=prompt)
                    ]
                },
                config={
                    "recursion_limit": 30
                }
            )
            answer = result["messages"][-1].content
            answer = answer.strip()
            if answer.startswith('"') and answer.endswith('"'):
                answer = answer[1:-1]
            return answer.strip()

        except Exception as e:
            return f"Agent error: {e}"

def run_and_submit_all( profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the BasicAgent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID") # Get the SPACE_ID for sending link to the code

    if profile:
        username= f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent ( modify this part to create your agent)
    try:
        agent = BasicAgent()
    except Exception as e:
        print(f"Error instantiating agent: {e}")
        return f"Error initializing agent: {e}", None
    # In the case of an app running as a hugging Face space, this link points toward your codebase ( usefull for others so please keep it public)
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    # 2. Fetch Questions
    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
             print("Fetched questions list is empty.")
             return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching questions: {e}")
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
         print(f"Error decoding JSON response from questions endpoint: {e}")
         print(f"Response text: {response.text[:500]}")
         return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        print(f"An unexpected error occurred fetching questions: {e}")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run your Agent
    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = agent(task_id, question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer})
        except Exception as e:
             print(f"Error running agent on task {task_id}: {e}")
             results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"})

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission 
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    # 5. Submit
    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        print(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1.  Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc ...
        2.  Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3.  Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        ---
        **Disclaimers:**
        Once clicking on the "submit button, it can take quite some time ( this is the time for the agent to go through all the questions).
        This space provides a basic setup and is intentionally sub-optimal to encourage you to develop your own, more robust solution. For instance for the delay process of the submit button, a solution could be to cache the answers and submit in a seperate action or even to answer the questions in async.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    # Removed max_rows=10 from DataFrame constructor
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table]
    )

if __name__ == "__main__":
    print("\n" + "-"*30 + " App Starting " + "-"*30)
    # Check for SPACE_HOST and SPACE_ID at startup for information
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID") # Get SPACE_ID at startup

    if space_host_startup:
        print(f"✅ SPACE_HOST found: {space_host_startup}")
        print(f"   Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("ℹ️  SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup: # Print repo URLs if SPACE_ID is found
        print(f"✅ SPACE_ID found: {space_id_startup}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"   Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("ℹ️  SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-"*(60 + len(" App Starting ")) + "\n")

    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)
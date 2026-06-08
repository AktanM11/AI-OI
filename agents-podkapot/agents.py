from llm import Agent
from tools import tool
from pydantic import BaseModel, Field



class RouterOutput(BaseModel):
    is_agent_needed: bool = Field(
        description="Whether another agent is required"
    )

    agent_task: str = Field(
        description="Task for the next agent"
    )

    response: str = Field(
        description="The final response to the user"
    )



Router = Agent(
    instructions="""
You are a routing agent.

If the user's request can be answered directly:
- is_agent_needed = false
- response = final answer
- agent_task = ""

If another agent is required:
- is_agent_needed = true
- agent_task = a clear task for the next agent
- response = intermediate response if necessary, otherwise ""

Be concise.
""",
    format=RouterOutput,
)



class CoderOutput(BaseModel):
    code: str = Field(
        description="The code to execute the task"
    )



Coder = Agent(
    instructions="""
Write executable Python code that solves the task and prints the final result.
Return only code.
""",
    format=CoderOutput,
)



import subprocess
import tempfile



def execute_python(code: str) -> str:

    with tempfile.NamedTemporaryFile(
        suffix=".py",
        mode="w",
        delete=False
    ) as f:

        f.write(code)
        path = f.name

    result = subprocess.run(
        ["python", path],
        capture_output=True,
        text=True,
        timeout=30
    )

    return result.stdout + result.stderr



@tool
def python_executor(task: str) -> str:
    """
    Generate and execute Python code to solve a task.

    Args:
        task: The task to solve.

    Returns:
        The execution result.
    """

    code = Coder(task).code
    result = execute_python(code)
    return result



tools = [
    {"type": "web_search"},
    python_executor.tool,
]


class RecurserOutput(BaseModel):
    is_task_achievable: bool = Field(
        description="Whether the task is achievable with the current tools"
    )
    is_task_completed: bool = Field(
        description="Whether the task is completed"
    )
    next_task: str = Field(
        description="The next task to complete the original task, if the task is not completed yet"
    )
    final_response: str = Field(
        description="The final response to the agent, if the task is completed"
    )



Recurser = Agent(
    instructions="""
Determine whether the task can be completed.

If impossible:
- is_task_achievable = false

If completed:
- is_task_completed = true
- final_response = result

Otherwise:
- is_task_completed = false
- next_task = next required task

Be concise.
""",
    format=RecurserOutput,
    tools=tools
)
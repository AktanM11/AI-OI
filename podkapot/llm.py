from openai import OpenAI
from pydantic import BaseModel
import json
import os
from dotenv import load_dotenv
load_dotenv()



client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)



class Agent(object):
    def __init__(
        self,
        instructions: str, 
        format: type[BaseModel],
        model: str = "gpt-5.5",
        tools: list = [],
    ):
        self.instructions = instructions
        self.format = format
        self.model = model
        self.schemas = [
            tool.schema if hasattr(tool, "schema") else tool
            for tool in tools
        ]
        self.tools = {tool.name: tool for tool in tools if hasattr(tool, "name")}
        
    def __call__(self, input: str) -> BaseModel:
        response = client.responses.parse(
            instructions=self.instructions,
            model=self.model,
            text_format=self.format,
            tools=self.schemas,
            input=input,
        )

        while True:
            if response.output_parsed is not None:
                return response.output_parsed
            
            function_calls = [
                item
                for item in response.output
                if item.type == "function_call"
            ]

            outputs = []

            for call in function_calls:
                tool = self.tools[call.name]
                args = json.loads(
                    call.arguments
                )

                print(f"Calling tool: {tool.name} with arguments: {args}")

                result = tool.function(**args)
                outputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": call.call_id,
                        "output": str(result),
                    }
                )

            response = client.responses.parse(
                instructions=self.instructions,
                model=self.model,
                text_format=self.format,
                tools=self.schemas,
                input=[
                    *response.output,
                    *outputs,
                ],
            )
from openai import OpenAI
from pydantic import BaseModel
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
        #memory: bool = False,
    ):
        self.instructions = instructions
        self.format = format
        self.model = model
        self.tools = tools
        #self.memory = memory
        
    def __call__(self, input: str) -> BaseModel:
        response = client.responses.parse(
            instructions=self.instructions,
            model=self.model,
            text_format=self.format,
            tools=self.tools,
            input=input,
        )
        return response.output_parsed
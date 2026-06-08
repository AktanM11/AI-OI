from dataclasses import dataclass
from typing import Callable



@dataclass
class Tool:
    name: str
    description: str
    schema: dict
    function: Callable

from inspect import signature
from typing import get_type_hints
from docstring_parser import parse



def python_to_json_type(t):
    mapping = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    return mapping.get(t, "string")



def tool(func):

    sig = signature(func)
    hints = get_type_hints(func)

    doc = parse(func.__doc__ or "")

    param_descriptions = {
        p.arg_name: p.description
        for p in doc.params
    }

    properties = {}
    required = []

    for name, param in sig.parameters.items():

        properties[name] = {
            "type": python_to_json_type(
                hints.get(name, str)
            ),
            "description": param_descriptions.get(
                name,
                ""
            )
        }

        if param.default is param.empty:
            required.append(name)

    schema = {
        "type": "function",
        "name": func.__name__,
        "description": doc.short_description or "",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required
        }
    }

    func.tool = Tool(
        name=func.__name__,
        description=doc.short_description or "",
        schema=schema,
        function=func
    )

    return func
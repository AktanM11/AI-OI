from agents import *

class Chat(object):
    def __init__(
        self,
        Router: Agent,
        Recurser: Agent,
    ):
        self.Router = Router
        self.Recurser = Recurser
        self.memory = []

    def __call__(self, input: str) -> str:
        self.memory.append(
            {
                'role': 'user',
                'content': input,
            }
        )

        router_output = self.Router(self.memory)

        self.memory.append(
            {
                'role': 'assistant',
                'content': router_output.response,
            }
        )

        print("Assistant:", router_output.response)

        if router_output.is_agent_needed:
            
            recurser_output = self.Recurser(router_output.agent_task)
        
            if recurser_output.is_task_achievable:
                while not recurser_output.is_task_completed:
                    self.memory.append(
                        {
                            'role': 'assistant',
                            'content': recurser_output.next_task,
                        }
                    )
                    print("Agent:", recurser_output.next_task)
                    recurser_output = self.Recurser(recurser_output.next_task)
                self.memory.append(
                    {
                        'role': 'assistant',
                        'content': recurser_output.final_response,
                    }
                )
                print("Agent:", recurser_output.final_response)
            else:
                self.memory.append(
                    {
                        'role': 'assistant',
                        'content': "The task is not achievable with the current tools.",
                    }
                )
                print("Agent: The task is not achievable with the current tools.")
                    
        
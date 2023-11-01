import requests
import time
import os

from typing import Any, List, Mapping, Optional

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM


link_on_endp = os.getenv("LINK_ON_ENDPOINT")

apikey = os.getenv("API_KEY_RUNPOD")
part_with_api = f"Bearer {apikey}"

headers =  {"Content-Type": "application/json", "Authorization": part_with_api}

template = """[INST] <<SYS>>
You are a respectful and honest assistant, answer the questions laconically and clearly. If you don't know the answer to a question, please don't share false information.
<</SYS>>
{request}[/INST]
"""

def generate_answer(request):
  prompt = {"input": 
            {"prompt": template.format(request=request),
             "max_new_tokens": 2500,
             "temperature": 0.9
            }
           }
  response = requests.post(f"{link_on_endp}/run", json=prompt, headers=headers)
  id_response = response.json()['id']

  get_api_url = f"{link_on_endp}/status/{id_response}"
  answer = requests.get(get_api_url, headers=headers)

  while answer.json()['status'] == 'IN_QUEUE' or answer.json()['status'] == 'IN_PROGRESS':
    time.sleep(3)
    answer = requests.get(get_api_url, headers=headers)

  return answer.json()['output']

class CustomLLM(LLM):
    n: int

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
          raise ValueError("stop kwargs are not permitted.")

        answer = generate_answer(prompt)

        return answer
        
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {"n": self.n}

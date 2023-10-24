# Test task


## Task statement
This test is designed to assess your skills in building, training, fine-tuning, and hosting a language model. Pick an open-source LLM of your choice (like Llama 2), host the model on the cloud, expose that as an API, upload a sample text file, and run a sample query based on the context in the input text file to demonstrate that it is working.

## Solution
The following technologies were used to approach the aforementioned problem:
1. Chat GPT from OpenAI as the LLM.
2. Streamlit as a framework to host the program on the cloud.
3. The code was added to a public github repo and then was built and hosted on cloud with streamlit.

## How to use
1. Open the [link](https://test-task.streamlit.app).
2. Upload your .txt file (one can try `sample.txt` from the repo).
3. Ask a question.

## Issues
In the following chapter I describe the issues, which are recommended to be improved
1. The biggest issue is the use of OpenAI chatGPT. It was done intentionally to accelerate the development of the test task. One of open-source models are suggested (Llama-2)
2. Loading the txt file is an overkill. It is recommended to upload a file and load it straight away without saving it to disk.

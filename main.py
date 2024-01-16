import gradio as gr
import os
from pathlib import Path
import shutil
import openai
import autogen
import chromadb
import multiprocessing as mp
from hiv-risk import assess_hiv_risk

from autogen.retrieve_utils import TEXT_FORMATS, get_file_from_url, is_url

config_list = [
    {
         'model':'gpt-4-1106-preview',
         'api_key' : 'xx'
     }]

llm_config={
        "request_timeout":600,
        "seed":42,
        "config_list":config_list,
        "temperature":0,
        "top_p":1

    }

TIMEOUT = 60

def initialize_agents(llm_config):
    if isinstance(llm_config, gr.State):
        _config_list = llm_config
    else:
        _config_list = llm_config

    autogen.ChatCompletion.start_logging()

    patients = autogen.UserProxyAgent(
        name="patients",
        code_execution_config={"last_n_messages": 2, "work_dir": "coding"},
        max_consecutive_auto_reply=0,
        is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="TERMINATE",
        llm_config=llm_config,
    )
    
    counselor = autogen.AssistantAgent(
        name="counselor",
        llm_config=llm_config,
        system_message="You are an HIV PrEP counselor. You will be guided by the HIV provider to assess patients' risk and conduct motivational interviewing for PrEP if indicated",
    )

    return patients, counselor



def get_description_text():
    return """
    # Chatbot for HIV Prevention and Action (CHIA)
    
    This demo shows how to use AI agent to conduct motivational interviewing to promote PrEP use.

    """


def initiate_chat_with_agent(problem, queue, n_results=3):
    try:
        counselor.initiate_chat(
            patients, problem=problem, silent=False, n_results=n_results
        )
        messages = counselor.chat_messages
        messages = [messages[k] for k in messages.keys()][0]
        messages = [m["content"] for m in messages if m["role"] == "user"]
        print("messages: ", messages)
    except Exception as e:
        messages = [str(e)]
    queue.put(messages)


def chatbot_reply(input_text):
    """Chat with the agent through terminal."""
    queue = mp.Queue()
    process = mp.Process(
        target=initiate_chat_with_agent,
        args=(input_text, queue),
    )
    process.start()
    try:
        # process.join(TIMEOUT+2)
        messages = queue.get(timeout=TIMEOUT)
    except Exception as e:
        messages = [
            str(e)
            if len(str(e)) > 0
            else "Invalid Request to OpenAI, please check your API keys."
        ]
    finally:
        try:
            process.terminate()
        except:
            pass
    return messages




with gr.Blocks() as demo:
   patients, counselor = initialize_agents(llm_config)

   gr.Markdown(get_description_text())
   chatbot = gr.Chatbot(
        [],
        elem_id="chatbot",
        bubble_full_width=False,
        #avatar_images=(None, (os.path.join(os.path.dirname(__file__), "autogen.png"))),
        # height=600,
   )

   txt_input = gr.Textbox(
        scale=4,
        show_label=False,
        placeholder="Enter text and press enter",
        container=False,
   )

   with gr.Row():


       clear = gr.ClearButton([txt_input, chatbot])




   txt_prompt = gr.Textbox(
         label="Enter your prompt for roles you want to talk with and press enter to replace the default prompt",
         max_lines=40,
         show_label=True,
         container=True,
         show_copy_button=True,
         #layout={"height": 20},
    )

   def respond(message, chat_history):

        config_list = llm_config
        messages = chatbot_reply(message)
        _msg = (
            messages[-1]
            if len(messages) > 0 and messages[-1] != "TERMINATE"
            else messages[-2]
            if len(messages) > 1
            else "Context is not enough for answering the question. Please press `enter` in the context url textbox to make sure the context is activated for the chat."
        )
        chat_history.append((message, _msg))
        return "", chat_history

   def update_prompt(prompt):
        counselor.customized_prompt = prompt
        return prompt


   txt_input.submit(
        respond,
        [txt_input, chatbot],
        [txt_input, chatbot],
        )
   txt_prompt.submit(update_prompt, [txt_prompt], [txt_prompt])


if __name__ == "__main__":
    demo.launch(share=True)

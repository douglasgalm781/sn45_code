import json
import time
import traceback
import bittensor as bt
from typing import Awaitable
from functools import partial
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
from coding.protocol import StreamCodeSynapse
from coding.helpers import chain_forward, string_forward


def parse_diff(diff_string):
    lines = diff_string.splitlines()
    file_diffs = {}
    current_file = None
    diff_content = []
    is_diff_block = False

    for line in lines:
        if "diff --git" in line:
            if current_file and diff_content:
                file_diffs[current_file] = "\n".join(diff_content)
            current_file = line.split()[-1]
            diff_content = []
            is_diff_block = False
        elif line.startswith("---") or line.startswith("+++"):
            # Ignore these lines, as they indicate the old/new file path
            continue
        elif line.startswith("@@"):
            is_diff_block = True
            continue
        elif is_diff_block:
            diff_content.append(line)

    if current_file and diff_content:
        file_diffs[current_file] = "\n".join(diff_content)

    return file_diffs


def miner_init(self):
    """
    Initializes the miner. This function is called once when the miner is created.
    """

    def model_factory(
        api_base="http://localhost:8000/v1",
        model_name=self.config.neuron.model_id,
        max_tokens=4096,
        temperature=0.7,
        top_p=1.0,
        chat=False,
    ):
        if chat:
            return ChatOpenAI(
                openai_api_base=api_base,
                openai_api_key="EMPTY",
                model_name=model_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                streaming=True,
            )
        return OpenAI(
            openai_api_base=api_base,
            openai_api_key="EMPTY",
            model_name=model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            streaming=True,
        )

    self.model_factory = model_factory

    self.model = model_factory(chat=True)
    self.mistral = model_factory(
        api_base="http://localhost:8001/v1",
        model_name="thesven/Mistral-7B-Instruct-v0.3-GPTQ",
        chat=True,
    )


def miner_process(self, synapse: StreamCodeSynapse) -> Awaitable:
    """
    The miner process function is called every time the miner receives a request. This function should contain the main logic of the miner.
    """

    if synapse.messages:
        query = synapse.messages[-1].content

    extra_info = {}
    stop = None
    chain = None
    chain_formatter = None
    query = synapse.query

    bt.logging.debug(f"📧 Query received, forwarding synapse: {synapse}")
    if "<|fim_hole|>" in synapse.query and not synapse.files:
        chain = self.model_factory(chat=False)
        chain_formatter = f"<fim_prefix>{synapse.query.replace('<|fim_hole|>', '<fim_suffix>')}<fim_middle>"
        stop = [
            "<fim_prefix>",
            "<fim_suffix>",
            "<fim_middle>",
            "//",
            "<｜end▁of▁sentence｜>",
            "\n\n",
            "\r\n\r\n",
            "/src/",
            "#- coding: utf-8",
            "```",
            "\ndef",
            "\nclass",
            '\n"""#',
        ]
    elif synapse.messages and synapse.files:
        chain = self.model
        for file in synapse.files:
            file.content = file.content.replace("}", "}}").replace("{", "{{")
            filestring += f"#{file.path}\n{file.content}\n"
        chain_formatter = synapse.messages + [
            {"role": "user", "content": f"{filestring}\n{synapse.query}"}
        ]
    elif synapse.messages:
        chain = self.model
        synapse.messages[0].role = "user"
        chain_formatter = [msg.dict() for msg in synapse.messages]
    elif "The following issue is:\n\n" in synapse.query:
        # this is a SWE-Bench style task
        prompt = synapse.query + "\n"
        for file in synapse.files:
            prompt += f"#Filename: {file.path}\n{file.content}\n"
        prompt += (
            "Respond only with the patch, only modify the files you have been provided."
        )
        model_res = (
            self.mistral.invoke([{"role": "user", "content": prompt[0:15000]}])
            .content.replace("<patch>", "")
            .replace("</patch>", "")
            .replace("b/", "")
            .replace("a/", "")
        )
        if "```" in model_res:
            model_res = model_res.split("```")[1]
        model_res = json.dumps(parse_diff(model_res))
        return synapse.create_streaming_response(partial(string_forward, model_res))
    elif synapse.files and "<|fim_hole|>" in synapse.query:
        chain = self.model_factory(chat=False)
        string = ""
        for file in synapse.files:
            if "path" not in file:
                file.path = ""
            string += f"<file_sep>{file.path}\n{file.content}\n"
        chain_formatter = (
            string
            + "<fim_prefix>"
            + synapse.query.replace("<|fim_hole|>", "<fim_middle>")
        )
    elif "write code to" in synapse.query:
        string = ""
        chain = self.mistral
        for file in synapse.files:
            if "path" not in file:
                file.path = ""
            string += f"{file.path}\n{file.content}\n"
        if string:
            "Using the above files, and responding only with python code \n"
        chain_formatter = string + synapse.query
    else:
        chain = self.model
        chain_formatter = synapse.query
    if stop:
        self.model = self.model.bind(stop=stop)
    if not chain:
        prompt = PromptTemplate.from_template("{query}")
        chain = prompt | self.model

    init_time = time.time()
    timeout_threshold = synapse.timeout

    streamer = partial(
        chain_forward,
        self,
        synapse.query,
        synapse.files,
        extra_info,
        init_time,
        timeout_threshold,
        chain,
        chain_formatter,
    )
    return synapse.create_streaming_response(streamer)

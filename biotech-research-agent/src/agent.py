"""
agent.py
---------
ReAct agent loop using Groq for LLM inference and tool calling.

Architecture: ReAct (Reason + Act)
  - The LLM reasons about which tool to call
  - We execute the tool and return the result
  - The LLM observes the result and decides: call another tool, or answer
  - Loop repeats until the LLM produces a final answer (no tool call)
  - Safety net: max_iterations prevents infinite loops

Groq tool calling is OpenAI-compatible:
  - Pass tool schemas in the `tools` parameter
  - When Groq wants to call a tool, it returns `finish_reason="tool_calls"`
  - We execute the tool and append the result as a `tool` message
  - Continue until `finish_reason="stop"`

Reference: Yao et al. (2022). ReAct: Synergizing Reasoning and Acting in LLMs.
"""

import os
import json
from dataclasses import dataclass, field
from typing import List, Optional

from groq import Groq
from loguru import logger

from src.tools import TOOL_SCHEMAS, call_tool


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class ToolCall:
    """Records a single tool invocation and its result."""
    tool_name: str
    arguments: dict
    result: str           # JSON string
    iteration: int


@dataclass
class AgentResult:
    """Full result from an agent run."""
    question: str
    answer: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    iterations: int = 0
    model: str = ""
    stopped_reason: str = "stop"  # "stop" | "max_iterations" | "error"

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "iterations": self.iterations,
            "tool_calls": [
                {
                    "tool": tc.tool_name,
                    "args": tc.arguments,
                    "result_preview": tc.result[:300],
                    "iteration": tc.iteration,
                }
                for tc in self.tool_calls
            ],
            "stopped_reason": self.stopped_reason,
        }


# ------------------------------------------------------------------
# Agent class
# ------------------------------------------------------------------

class BiotechResearchAgent:
    """
    Agentic biotech researcher powered by Groq + 6 scientific database tools.

    Usage:
        agent = BiotechResearchAgent.from_config("config.yaml")
        result = agent.run("What phase 3 trials exist for KRAS G12C inhibitors?")
        print(result.answer)
    """

    def __init__(
        self,
        model: str = "llama3-70b-8192",
        system_prompt: str = "",
        max_iterations: int = 8,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ):
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not set. Get a free key at console.groq.com "
                "and add it to your .env file."
            )

        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.system_prompt = system_prompt or _default_system_prompt()

        logger.info(f"[agent] Initialized: model={model}, max_iter={max_iterations}")

    @classmethod
    def from_config(cls, config_path: str = "config.yaml") -> "BiotechResearchAgent":
        """Load agent settings from config.yaml."""
        import yaml
        from pathlib import Path
        config = yaml.safe_load(Path(config_path).read_text())
        llm = config["llm"]
        agent_cfg = config["agent"]
        return cls(
            model=llm["model"],
            system_prompt=agent_cfg.get("system_prompt", ""),
            max_iterations=agent_cfg.get("max_iterations", 8),
            temperature=llm.get("temperature", 0.1),
            max_tokens=llm.get("max_tokens", 2048),
        )

    def run(self, question: str) -> AgentResult:
        """
        Run the ReAct agent loop on a question.

        Parameters
        ----------
        question : str
            The user's research question.

        Returns
        -------
        AgentResult with final answer + full tool call trace.
        """
        logger.info(f"[agent] Question: {question[:80]}...")

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": question},
        ]

        tool_calls_log: List[ToolCall] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.debug(f"[agent] Iteration {iteration}/{self.max_iterations}")

            # Call Groq
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=TOOL_SCHEMAS,
                    tool_choice="auto",
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
            except Exception as e:
                logger.error(f"[agent] Groq API error: {e}")
                return AgentResult(
                    question=question,
                    answer=f"⚠️ API error: {e}",
                    tool_calls=tool_calls_log,
                    iterations=iteration,
                    model=self.model,
                    stopped_reason="error",
                )

            choice = response.choices[0]
            finish_reason = choice.finish_reason
            message = choice.message

            # Append assistant message
            messages.append(message.model_dump(exclude_unset=True))

            # --------------------------------------------------
            # Case 1: LLM wants to call tools
            # --------------------------------------------------
            if finish_reason == "tool_calls" and message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.function.name
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    logger.info(f"[agent] → Tool: {tool_name}({tool_args})")

                    # Execute the tool
                    result_json = call_tool(tool_name, tool_args)

                    # Log the call
                    tool_calls_log.append(ToolCall(
                        tool_name=tool_name,
                        arguments=tool_args,
                        result=result_json,
                        iteration=iteration,
                    ))

                    # Append tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_json,
                    })

            # --------------------------------------------------
            # Case 2: LLM has finished — return final answer
            # --------------------------------------------------
            elif finish_reason == "stop":
                answer = message.content or ""
                logger.info(
                    f"[agent] Done after {iteration} iterations, "
                    f"{len(tool_calls_log)} tool calls"
                )
                return AgentResult(
                    question=question,
                    answer=answer,
                    tool_calls=tool_calls_log,
                    iterations=iteration,
                    model=self.model,
                    stopped_reason="stop",
                )

            else:
                # Unexpected finish reason
                logger.warning(f"[agent] Unexpected finish_reason: {finish_reason}")
                break

        # Max iterations reached — ask Groq to wrap up
        logger.warning(f"[agent] Max iterations ({self.max_iterations}) reached. Forcing answer.")
        messages.append({
            "role": "user",
            "content": (
                "Please now provide your best answer based on everything you've "
                "gathered so far. Be clear about what you found and any gaps."
            ),
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = response.choices[0].message.content or "Unable to generate answer."
        except Exception as e:
            answer = f"Max iterations reached. Last tool calls: {[tc.tool_name for tc in tool_calls_log]}"

        return AgentResult(
            question=question,
            answer=answer,
            tool_calls=tool_calls_log,
            iterations=self.max_iterations,
            model=self.model,
            stopped_reason="max_iterations",
        )


def _default_system_prompt() -> str:
    return """You are an expert biotech and pharmaceutical research assistant.
You have access to six scientific databases:
  - search_pubmed: peer-reviewed literature (PubMed)
  - search_trials: clinical trials (ClinicalTrials.gov)
  - lookup_protein: protein/gene function (UniProt)
  - query_drug_targets: drug-target-disease associations (Open Targets)
  - lookup_drug: compound info and mechanism of action (ChEMBL)
  - search_genes: gene information (NCBI Gene)

Guidelines:
1. Use multiple tools when the question spans different domains.
2. Always gather evidence before answering — don't guess.
3. For drug questions: use both ChEMBL (mechanism) and Open Targets (clinical stage).
4. For disease questions: combine PubMed (evidence) + ClinicalTrials (trial landscape).
5. Cite sources: include PubMed IDs, NCT numbers, UniProt accessions.
6. Be precise: distinguish approved vs. investigational, clinical vs. preclinical.
7. If you cannot find sufficient evidence, state this clearly.
8. Structure your answer with clear sections when the answer is complex."""

#!/usr/bin/env python3
"""
对虾育种智能决策系统 — Gradio 多轮对话界面
Shrimp Breeding AI Decision System — Gradio Chat Interface

基于 shrimp_breeding_rag_agent.py，增加 Gradio Web 聊天界面，
支持多轮对话、上下文理解和流式输出。

启动方式：
    source .venv/bin/activate
    set -a && source .env && set +a
    python examples/shrimp_breeding_chat.py
"""

from __future__ import annotations

import json
import os
from typing import Any, Generator

import gradio as gr

# 复用原有模块的核心组件
from shrimp_breeding_rag_agent import (
    SYSTEM_PROMPT,
    TOOLS,
    build_client,
    dispatch_tool,
)

# ---------------------------------------------------------------------------
# 全局客户端
# ---------------------------------------------------------------------------
_provider = os.environ.get("LLM_PROVIDER", "openai")
_client, _model = build_client(_provider)


# ---------------------------------------------------------------------------
# 核心：多轮对话 Agent（流式输出）
# ---------------------------------------------------------------------------

def _build_messages(
    history: list[dict[str, str]],
    user_input: str,
) -> list[dict[str, Any]]:
    """将 Gradio 聊天历史 + 新用户输入转换为 OpenAI messages 格式。"""
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    # Gradio 的 type="messages" 格式：[{"role": "user"/"assistant", "content": "..."}]
    for msg in history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": user_input})
    return messages


def chat_agent(
    user_input: str,
    history: list[dict[str, str]],
) -> Generator[str, None, None]:
    """
    Gradio ChatInterface 的回调函数。
    执行 Agent 循环（工具调用），最终流式返回 LLM 的回答。
    """
    if not user_input.strip():
        yield "请输入您的育种问题。"
        return

    messages = _build_messages(history, user_input)
    max_iterations = 10

    # --- Agent 工具调用循环（非流式，因为需要处理 tool_calls）---
    for _ in range(max_iterations):
        response = _client.chat.completions.create(
            model=_model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )

        choice = response.choices[0]
        message = choice.message

        # 构建 assistant 消息
        assistant_msg: dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
        }
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        # 没有工具调用 → 进入流式输出最终回答
        if not message.tool_calls:
            break

        # 执行工具调用，结果追加到 messages
        for tc in message.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            tool_result = dispatch_tool(fn_name, fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": tool_result,
            })

    # --- 流式输出最终回答 ---
    # 如果上面的循环已经拿到了非工具调用的回答，再用流式重新生成
    # 去掉最后一条 assistant 消息，改为流式生成
    if messages[-1]["role"] == "assistant" and "tool_calls" not in messages[-1]:
        messages.pop()

    stream = _client.chat.completions.create(
        model=_model,
        messages=messages,
        temperature=0.2,
        stream=True,
    )

    accumulated = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            accumulated += chunk.choices[0].delta.content
            yield accumulated


# ---------------------------------------------------------------------------
# Gradio 界面
# ---------------------------------------------------------------------------

EXAMPLES = [
    "我想提高对虾生长速度，同时保证抗WSSV能力不下降，请推荐交配方案。",
    "请评估家系F001、F003、F006的遗传多样性状况，是否存在近交风险？",
    "什么是BLUP育种值估计？在对虾家系育种中如何应用？",
    "查询所有家系的性能数据，哪些家系抗病性最强？",
    "如何利用基因组选择技术提高选择准确性？",
]

demo = gr.ChatInterface(
    fn=chat_agent,
    title="🦐 对虾育种智能决策系统",
    description=(
        "基于 RAG + Agent 架构，支持育种知识检索、家系性能查询、"
        "交配方案推荐和遗传多样性评估。多轮对话，理解上下文。\n\n"
        f"**当前配置**：LLM = `{_provider}` / `{_model}` | 可用家系 = F001–F006"
    ),
    examples=EXAMPLES,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(
        placeholder="例如：推荐抗WSSV能力最强的交配组合",
        container=False,
        scale=7,
    ),
    save_history=True,
)

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
    )

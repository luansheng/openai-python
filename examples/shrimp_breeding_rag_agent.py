#!/usr/bin/env python3
"""
对虾育种 RAG + 智能体决策系统示例
Shrimp Aquaculture Family Breeding — RAG + Agentic Decision Platform

This example demonstrates how to build a Retrieval-Augmented Generation (RAG) +
intelligent-agent platform for aquaculture shrimp family selective breeding using the
openai-python SDK.  Because the SDK accepts any OpenAI-compatible endpoint, you can
swap the underlying LLM provider by changing `base_url` and `api_key` — no other code
change is required.

Supported providers (non-exhaustive):
  - OpenAI          — https://api.openai.com/v1                (OPENAI_API_KEY)
  - DeepSeek        — https://api.deepseek.com/v1              (DEEPSEEK_API_KEY)
  - Moonshot (Kimi) — https://api.moonshot.cn/v1               (MOONSHOT_API_KEY)
  - Alibaba Tongyi  — https://dashscope.aliyuncs.com/compatible-mode/v1  (DASHSCOPE_API_KEY)
  - Zhipu GLM       — https://open.bigmodel.cn/api/paas/v4     (ZHIPU_API_KEY)
  - Baichuan        — https://api.baichuan-ai.com/v1           (BAICHUAN_API_KEY)
  - 01.AI Yi        — https://api.lingyiwanwu.com/v1           (YI_API_KEY)

Usage
-----
Set the relevant environment variables for your chosen provider and run:

    python examples/shrimp_breeding_rag_agent.py

Or override the provider programmatically before calling `run_breeding_agent()`.

Architecture Overview
---------------------
1. **Knowledge Base** — A small in-memory corpus of shrimp-breeding documents
   (family records, genetic-diversity metrics, disease-resistance scores, growth
   performance, water-quality guidelines, etc.).

2. **RAG Retrieval** — A lightweight TF-IDF-style keyword retriever selects the
   most relevant documents for a given query without requiring an external vector
   database, making the example fully self-contained.  In production you would
   replace this with a dedicated vector store (e.g. Chroma, Milvus, Weaviate)
   backed by an embedding model.

3. **Agent Tools** — Four function-call tools are exposed to the LLM:
     • retrieve_breeding_knowledge  — retrieves relevant passages from the KB
     • query_family_performance     — returns mock EBV / growth metrics per family
     • recommend_cross_pairs        — suggests optimal parent pairings
     • evaluate_genetic_diversity   — returns diversity / inbreeding coefficients

4. **Agent Loop** — The orchestrator sends the user query to the LLM, executes
   requested tool calls, feeds results back, and repeats until the model returns
   a final answer with no further tool calls.

5. **Streaming** — The final answer is streamed token-by-token to stdout so the
   user sees the response immediately.
"""

from __future__ import annotations

import os
import re
import json
import math
from typing import Any

from openai import OpenAI

# ---------------------------------------------------------------------------
# 1.  Provider configuration
# ---------------------------------------------------------------------------

# Select your provider by setting the LLM_PROVIDER environment variable, e.g.:
#   export LLM_PROVIDER=deepseek
# Defaults to "openai".

_PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "model": "gpt-4o",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "api_key_env": "MOONSHOT_API_KEY",
        "model": "moonshot-v1-8k",
    },
    "tongyi": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "model": "qwen-max",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "api_key_env": "ZHIPU_API_KEY",
        "model": "glm-4",
    },
    "baichuan": {
        "base_url": "https://api.baichuan-ai.com/v1",
        "api_key_env": "BAICHUAN_API_KEY",
        "model": "Baichuan4",
    },
    "yi": {
        "base_url": "https://api.lingyiwanwu.com/v1",
        "api_key_env": "YI_API_KEY",
        "model": "yi-large",
    },
}


def build_client(provider: str = "openai") -> tuple[OpenAI, str]:
    """Return an OpenAI-compatible client and model name for *provider*."""
    cfg = _PROVIDERS.get(provider)
    if cfg is None:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")

    api_key = os.environ.get(cfg["api_key_env"])
    if not api_key:
        raise EnvironmentError(
            f"API key not found. Please set the environment variable '{cfg['api_key_env']}' "
            f"for provider '{provider}'."
        )
    client = OpenAI(
        api_key=api_key,
        base_url=cfg["base_url"],
    )
    return client, cfg["model"]


# ---------------------------------------------------------------------------
# 2.  In-memory knowledge base (shrimp breeding domain)
# ---------------------------------------------------------------------------

# Maximum number of characters included in each knowledge-base snippet sent to the LLM.
MAX_CONTENT_LENGTH = 400

# Maximum number of characters shown in the verbose tool-result preview line.
PREVIEW_MAX_LENGTH = 200

_KNOWLEDGE_BASE: list[dict[str, str]] = [
    {
        "id": "kb_001",
        "title": "对虾家系选育概述",
        "content": (
            "对虾家系选育（Family Selective Breeding）是以家系为基本遗传单元，通过系统评估"
            "家系间遗传性状差异，筛选优良亲本进行定向交配的育种方法。"
            "主要目标性状包括：生长速度（体重、体长）、抗病性（白斑综合征病毒WSSV、"
            "急性肝胰腺坏死病AHPND）、饲料转化率和存活率。"
            "育种流程：亲本选择→人工授精→孵化管理→幼虾培育→性状测定→BLUP估计育种值→"
            "选留优良个体→建立新家系。"
        ),
    },
    {
        "id": "kb_002",
        "title": "最佳线性无偏预测（BLUP）育种值估计",
        "content": (
            "BLUP（Best Linear Unbiased Prediction）是水产家系育种中估计育种值（EBV）的"
            "标准统计方法。混合模型方程为：y = Xb + Zu + e，"
            "其中 y 为表型观测值，b 为固定效应（如养殖批次、水温），"
            "u 为随机加性遗传效应，e 为残差。"
            "遗传参数（遗传力 h²、遗传相关）通过 REML 法估计。"
            "对虾生长性状遗传力通常在 0.15–0.40 之间，抗病性状遗传力为 0.10–0.25。"
            "常用软件：ASReml、WOMBAT、DMU、blupf90。"
        ),
    },
    {
        "id": "kb_003",
        "title": "遗传多样性与近交管理",
        "content": (
            "近交系数（F）反映个体因亲缘交配导致的纯合度增加程度。"
            "封闭育种群每代近交增量（ΔF）应控制在 0.5%–1% 以内，以维持长期遗传进展。"
            "管理策略：①轮回交配设计（Rotational Mating）；"
            "②最优贡献选择（Optimal Contribution Selection, OCS）；"
            "③引入外来亲本扩大遗传基础；"
            "④基因组选择（Genomic Selection）利用SNP芯片追踪祖先贡献。"
            "遗传多样性指标：观测杂合度（Ho）、期望杂合度（He）、等位基因丰富度（AR）。"
        ),
    },
    {
        "id": "kb_004",
        "title": "对虾抗病性状选育",
        "content": (
            "白斑综合征病毒（WSSV）是对虾养殖中危害最严重的病原，"
            "抗WSSV性状采用存活分析（Survival Analysis）方法评估。"
            "攻毒试验设计：将家系幼虾暴露于已知剂量WSSV，记录48–96小时存活率。"
            "急性肝胰腺坏死病（AHPND）由弧菌（Vibrio parahaemolyticus）引起，"
            "可通过人工攻毒或基因组标记辅助选择（MAS）筛选抗性个体。"
            "建议抗病选择指数权重：生长性状 60%，抗WSSV 25%，抗AHPND 15%。"
        ),
    },
    {
        "id": "kb_005",
        "title": "对虾繁殖技术与孵化管理",
        "content": (
            "南美白对虾（Litopenaeus vannamei）人工繁殖流程："
            "①亲虾暂养驯化（水温 26–28°C，盐度 30–35‰）；"
            "②眼柄切除促进性腺发育（雌虾）；"
            "③交配：雄虾精荚移植至雌虾纳精囊；"
            "④产卵：水温升至 28–30°C 诱导产卵，每尾产卵 20–50 万粒；"
            "⑤孵化：无节幼体（Nauplii）12–15 小时孵出；"
            "⑥幼体培育：无节幼体→蚤状幼体→糠虾幼体→仔虾（约 12 天）；"
            "⑦仔虾标记：荧光染料或PIT标签用于家系溯源。"
        ),
    },
    {
        "id": "kb_006",
        "title": "基因组选择与SNP芯片应用",
        "content": (
            "基因组选择（GS）利用全基因组SNP标记估计基因组育种值（GEBV），"
            "可大幅提高选择准确性，缩短世代间隔。"
            "对虾基因组选择流程：①构建参考群体（已测定表型+基因型）；"
            "②选择合适统计模型（GBLUP、BayesB、RKHS等）；"
            "③验证预测准确性（交叉验证）；"
            "④对候选个体进行基因分型，预测GEBV；"
            "⑤结合传统EBV构建综合选择指数。"
            "常用SNP密度：南美白对虾参考基因组约 2.4 Gb，"
            "推荐使用 50K–200K SNP芯片。"
        ),
    },
    {
        "id": "kb_007",
        "title": "水质管理与养殖环境优化",
        "content": (
            "对虾最适养殖水质参数：水温 23–30°C，盐度 10–35‰，"
            "pH 7.5–8.5，溶氧 ≥ 5 mg/L，氨氮 < 0.5 mg/L，亚硝酸盐 < 0.1 mg/L，"
            "硬度（总碱度）100–150 mg/L。"
            "超高密度养殖（> 200尾/m²）需配备增氧机、生物絮团系统或RAS循环水系统。"
            "益生菌（芽孢杆菌、乳酸菌）可改善肠道菌群，提升免疫力，减少抗生素用量。"
        ),
    },
    {
        "id": "kb_008",
        "title": "RAG系统在育种决策中的应用",
        "content": (
            "检索增强生成（RAG）系统通过将大型语言模型（LLM）与外部知识库结合，"
            "可显著提升育种决策的准确性和可解释性。"
            "典型架构：①文档向量化存储（Embedding + Vector DB）；"
            "②查询向量化与相似度检索（Cosine Similarity / BM25）；"
            "③上下文拼接生成（Prompt Engineering）；"
            "④结构化输出（JSON Schema / Function Calling）。"
            "在对虾育种中，RAG可快速检索文献数据库、家系档案、疾病诊断记录，"
            "辅助育种师制定选育方案、评估风险、预测遗传进展。"
        ),
    },
]


# ---------------------------------------------------------------------------
# 3.  Lightweight keyword-based retriever (no external dependencies)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Very simple tokenizer: split on whitespace and punctuation."""
    return re.findall(r"[\w\u4e00-\u9fff]+", text.lower())


def _tf_idf_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """
    Compute a simple overlap score between query and document token sets.
    Weights repeated query terms and document term frequency.
    """
    doc_freq: dict[str, int] = {}
    for t in doc_tokens:
        doc_freq[t] = doc_freq.get(t, 0) + 1
    score = 0.0
    for qt in set(query_tokens):
        if qt in doc_freq:
            # term frequency in doc, log-normalized
            score += 1 + math.log(doc_freq[qt])
    return score


def retrieve_documents(query: str, top_k: int = 3) -> list[dict[str, str]]:
    """Return the *top_k* most relevant knowledge-base documents for *query*."""
    q_tokens = _tokenize(query)
    scored: list[tuple[float, dict[str, str]]] = []
    for doc in _KNOWLEDGE_BASE:
        doc_text = doc["title"] + " " + doc["content"]
        score = _tf_idf_score(q_tokens, _tokenize(doc_text))
        scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]


# ---------------------------------------------------------------------------
# 4.  Mock breeding-database functions (replace with real DB calls)
# ---------------------------------------------------------------------------

_FAMILY_DB: dict[str, dict[str, Any]] = {
    "F001": {"name": "家系F001", "ebv_growth": 12.5, "ebv_survival": 8.3, "inbreeding": 0.02, "wssv_resistance": 0.78},
    "F002": {"name": "家系F002", "ebv_growth": 9.8,  "ebv_survival": 14.1, "inbreeding": 0.01, "wssv_resistance": 0.85},
    "F003": {"name": "家系F003", "ebv_growth": 15.2, "ebv_survival": 6.7,  "inbreeding": 0.03, "wssv_resistance": 0.62},
    "F004": {"name": "家系F004", "ebv_growth": 11.0, "ebv_survival": 11.0, "inbreeding": 0.01, "wssv_resistance": 0.80},
    "F005": {"name": "家系F005", "ebv_growth": 7.5,  "ebv_survival": 16.2, "inbreeding": 0.00, "wssv_resistance": 0.91},
    "F006": {"name": "家系F006", "ebv_growth": 13.8, "ebv_survival": 10.5, "inbreeding": 0.02, "wssv_resistance": 0.74},
}


def _query_family_performance(family_ids: list[str]) -> dict[str, Any]:
    """Return EBV and phenotypic summary for a list of family IDs."""
    results: dict[str, Any] = {}
    for fid in family_ids:
        if fid in _FAMILY_DB:
            results[fid] = _FAMILY_DB[fid]
        else:
            results[fid] = {"error": f"家系 {fid} 不存在"}
    return results


def _recommend_cross_pairs(
    objective: str,
    growth_weight: float = 0.5,
    survival_weight: float = 0.3,
    resistance_weight: float = 0.2,
    max_inbreeding: float = 0.03,
    top_n: int = 3,
) -> list[dict[str, Any]]:
    """
    Suggest top-N parent cross pairs based on a weighted selection index.
    Avoids crosses where both parents share high inbreeding coefficients.
    """
    families = list(_FAMILY_DB.values())
    family_ids = list(_FAMILY_DB.keys())

    # Normalize EBV columns to [0, 1]
    def normalize(vals: list[float]) -> list[float]:
        mn, mx = min(vals), max(vals)
        return [(v - mn) / (mx - mn) if mx > mn else 0.5 for v in vals]

    n_growth = normalize([f["ebv_growth"] for f in families])
    n_survival = normalize([f["ebv_survival"] for f in families])
    n_resistance = normalize([f["wssv_resistance"] for f in families])

    # Compute index for each family
    indices = [
        growth_weight * g + survival_weight * s + resistance_weight * r
        for g, s, r in zip(n_growth, n_survival, n_resistance)
    ]

    # Rank pairs by mean index, filter on inbreeding
    pairs: list[tuple[float, str, str]] = []
    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            if families[i]["inbreeding"] + families[j]["inbreeding"] <= max_inbreeding * 2:
                pair_score = (indices[i] + indices[j]) / 2
                pairs.append((pair_score, family_ids[i], family_ids[j]))

    pairs.sort(reverse=True)
    recommendations = []
    for score, fid_a, fid_b in pairs[:top_n]:
        recommendations.append(
            {
                "female_family": fid_a,
                "male_family": fid_b,
                "selection_index": round(score, 4),
                "expected_ebv_growth": round(
                    (_FAMILY_DB[fid_a]["ebv_growth"] + _FAMILY_DB[fid_b]["ebv_growth"]) / 2, 2
                ),
                "objective": objective,
            }
        )
    return recommendations


def _evaluate_genetic_diversity(family_ids: list[str]) -> dict[str, Any]:
    """Return genetic diversity metrics for the given set of families."""
    selected = [_FAMILY_DB[f] for f in family_ids if f in _FAMILY_DB]
    if not selected:
        return {"error": "没有找到有效家系"}
    avg_inbreeding = sum(f["inbreeding"] for f in selected) / len(selected)
    delta_f = avg_inbreeding  # simplified: assume one generation
    mean_resistance = sum(f["wssv_resistance"] for f in selected) / len(selected)
    return {
        "n_families": len(selected),
        "avg_inbreeding_coefficient": round(avg_inbreeding, 4),
        "delta_F_per_generation": round(delta_f, 4),
        "mean_wssv_resistance": round(mean_resistance, 4),
        "diversity_status": "良好" if delta_f < 0.01 else ("警告" if delta_f < 0.02 else "危险"),
    }


# ---------------------------------------------------------------------------
# 5.  Tool definitions (OpenAI function-calling schema)
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_breeding_knowledge",
            "description": (
                "从对虾育种专业知识库中检索与查询最相关的文档段落，"
                "用于回答育种方法、遗传参数、疾病防控等问题。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询文本，应尽量精确描述所需信息",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回最相关文档数量，默认为 3",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_family_performance",
            "description": "查询指定对虾育种家系的遗传育种值（EBV）、近交系数和抗病性数据。",
            "parameters": {
                "type": "object",
                "properties": {
                    "family_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "家系编号列表，例如 [\"F001\", \"F002\"]",
                    }
                },
                "required": ["family_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recommend_cross_pairs",
            "description": (
                "根据育种目标和选择指数权重，推荐最优亲本交配组合，"
                "同时考虑近交系数限制以维持遗传多样性。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "objective": {
                        "type": "string",
                        "description": "育种目标描述，例如 '提高生长速度兼顾抗WSSV'",
                    },
                    "growth_weight": {
                        "type": "number",
                        "description": "生长性状权重 (0–1)，默认 0.5",
                        "default": 0.5,
                    },
                    "survival_weight": {
                        "type": "number",
                        "description": "存活性状权重 (0–1)，默认 0.3",
                        "default": 0.3,
                    },
                    "resistance_weight": {
                        "type": "number",
                        "description": "抗病性权重 (0–1)，默认 0.2",
                        "default": 0.2,
                    },
                    "max_inbreeding": {
                        "type": "number",
                        "description": "单亲最大允许近交系数，默认 0.03",
                        "default": 0.03,
                    },
                    "top_n": {
                        "type": "integer",
                        "description": "返回推荐交配组合数量，默认 3",
                        "default": 3,
                    },
                },
                "required": ["objective"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "evaluate_genetic_diversity",
            "description": "评估所选家系群体的遗传多样性，包括近交系数、每代近交增量和抗病性均值。",
            "parameters": {
                "type": "object",
                "properties": {
                    "family_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "参与评估的家系编号列表",
                    }
                },
                "required": ["family_ids"],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# 6.  Tool dispatcher
# ---------------------------------------------------------------------------

def dispatch_tool(name: str, args: dict[str, Any]) -> str:
    """Execute the tool identified by *name* and return a JSON string result."""
    if name == "retrieve_breeding_knowledge":
        docs = retrieve_documents(args["query"], top_k=args.get("top_k", 3))
        snippets = [
            {"id": d["id"], "title": d["title"], "content": d["content"][:MAX_CONTENT_LENGTH]}
            for d in docs
        ]
        return json.dumps(snippets, ensure_ascii=False, indent=2)

    elif name == "query_family_performance":
        result = _query_family_performance(args["family_ids"])
        return json.dumps(result, ensure_ascii=False, indent=2)

    elif name == "recommend_cross_pairs":
        result = _recommend_cross_pairs(
            objective=args["objective"],
            growth_weight=args.get("growth_weight", 0.5),
            survival_weight=args.get("survival_weight", 0.3),
            resistance_weight=args.get("resistance_weight", 0.2),
            max_inbreeding=args.get("max_inbreeding", 0.03),
            top_n=args.get("top_n", 3),
        )
        return json.dumps(result, ensure_ascii=False, indent=2)

    elif name == "evaluate_genetic_diversity":
        result = _evaluate_genetic_diversity(args["family_ids"])
        return json.dumps(result, ensure_ascii=False, indent=2)

    else:
        return json.dumps({"error": f"未知工具: {name}"})


# ---------------------------------------------------------------------------
# 7.  System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
你是一位专业的对虾水产家系育种智能决策助手，具备以下能力：

1. **RAG知识检索**：能够从对虾育种专业知识库中检索相关信息，包括选育方法、
   遗传参数估计（BLUP/GBLUP）、抗病性评价、基因组选择等。

2. **家系数据查询**：可查询各育种家系的遗传育种值（EBV）、近交系数、
   抗病性指标等关键参数。

3. **交配方案推荐**：根据育种目标和遗传参数，推荐最优亲本交配组合，
   同时控制近交水平以维持群体遗传多样性。

4. **遗传多样性评估**：评估选育群体的遗传健康状况，预警近交风险。

工作原则：
- 每次决策前先检索相关专业知识，确保回答有据可依
- 提供结构化的分析报告，包括数据依据和建议理由
- 平衡遗传进展与遗传多样性保护
- 用中文回答，专业术语后附英文注释

可用家系编号：F001, F002, F003, F004, F005, F006
"""

# ---------------------------------------------------------------------------
# 8.  Agent loop
# ---------------------------------------------------------------------------

def run_breeding_agent(
    user_query: str,
    provider: str = "openai",
    max_iterations: int = 10,
    verbose: bool = True,
) -> str:
    """
    Run the shrimp-breeding RAG + agent loop for *user_query*.

    Parameters
    ----------
    user_query:
        The breeding question or decision request from the user.
    provider:
        LLM provider key (see _PROVIDERS).  Override with LLM_PROVIDER env var.
    max_iterations:
        Maximum number of LLM ↔ tool call cycles before forcing a stop.
    verbose:
        Print tool calls and intermediate steps to stdout.

    Returns
    -------
    The final textual answer from the LLM.
    """
    provider = os.environ.get("LLM_PROVIDER", provider)
    client, model = build_client(provider)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    final_answer = ""

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n[迭代 {iteration + 1}] 调用 LLM ({model})…")

        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )

        choice = response.choices[0]
        message = choice.message

        # Append assistant turn (must include tool_calls if present)
        assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
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

        # No tool calls → final answer
        if not message.tool_calls:
            final_answer = message.content or ""
            break

        # Execute tool calls
        for tc in message.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)

            if verbose:
                print(f"  → 工具调用: {fn_name}({json.dumps(fn_args, ensure_ascii=False)})")

            tool_result = dispatch_tool(fn_name, fn_args)

            if verbose:
                # Print a brief preview of the result
                preview = tool_result[:PREVIEW_MAX_LENGTH].replace("\n", " ")
                print(f"  ← 工具结果: {preview}…")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result,
                }
            )
    else:
        # Exceeded max iterations — get a final response without tools
        if verbose:
            print("\n[达到最大迭代次数，强制获取最终回答]")
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.2,
        )
        final_answer = response.choices[0].message.content or ""

    return final_answer


# ---------------------------------------------------------------------------
# 9.  Streaming variant
# ---------------------------------------------------------------------------

def run_breeding_agent_streaming(
    user_query: str,
    provider: str = "openai",
    max_iterations: int = 10,
) -> None:
    """
    Same as run_breeding_agent but streams the final LLM answer token-by-token.
    Tool-call iterations are executed non-streaming for simplicity.
    """
    provider = os.environ.get("LLM_PROVIDER", provider)
    client, model = build_client(provider)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query},
    ]

    for _ in range(max_iterations):
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.2,
        )

        choice = response.choices[0]
        message = choice.message

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in message.tool_calls
            ]
        messages.append(assistant_msg)

        if not message.tool_calls:
            # Stream the final answer
            print("\n\n══════════════ 育种决策报告（流式输出）══════════════\n")
            stream = client.chat.completions.create(
                model=model,
                messages=messages[:-1],  # re-generate the last assistant turn with streaming
                temperature=0.2,
                stream=True,
            )
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    print(chunk.choices[0].delta.content, end="", flush=True)
            print("\n")
            return

        for tc in message.tool_calls:
            fn_name = tc.function.name
            fn_args = json.loads(tc.function.arguments)
            print(f"[工具] {fn_name} → ", end="", flush=True)
            tool_result = dispatch_tool(fn_name, fn_args)
            print("完成")
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": tool_result})


# ---------------------------------------------------------------------------
# 10.  Demo entry-point
# ---------------------------------------------------------------------------

_DEMO_QUERIES = [
    (
        "综合分析",
        "我想在下一个育种世代中重点提高对虾生长速度，同时保证抗白斑综合征病毒（WSSV）能力不下降。"
        "请检索相关育种知识，查询所有家系的性能数据，推荐最优交配组合，并评估选留家系的遗传多样性。",
    ),
    (
        "近交风险评估",
        "请评估当前家系F001、F003、F006的遗传多样性状况，并告诉我是否存在近交风险。",
    ),
    (
        "知识问答",
        "什么是BLUP育种值估计？在对虾家系育种中如何应用？",
    ),
]


def main() -> None:
    print("=" * 60)
    print("  对虾水产家系育种 RAG + 智能体决策系统")
    print("  Shrimp Aquaculture Breeding RAG+Agent Platform")
    print("=" * 60)

    provider = os.environ.get("LLM_PROVIDER", "openai")
    print(f"\n使用的 LLM 提供商: {provider}")
    print("（通过环境变量 LLM_PROVIDER 切换: openai / deepseek / moonshot / tongyi / zhipu …）\n")

    # Run the first demo query with full verbosity
    title, query = _DEMO_QUERIES[0]
    print(f"\n{'─' * 60}")
    print(f"示例查询 [{title}]:")
    print(f"  {query}")
    print(f"{'─' * 60}")

    answer = run_breeding_agent(query, provider=provider, verbose=True)

    print("\n\n══════════════ 育种决策报告 ══════════════\n")
    print(answer)
    print("\n══════════════════════════════════════════\n")


if __name__ == "__main__":
    main()

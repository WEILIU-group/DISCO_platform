"""Dash-based chat UI for DISCO-Pilot.

Reimplements the previous Streamlit interface using Dash so it fits with
the rest of the platform modules. The flow remains the same:

1) User enters a research request and clicks "Generate Plan".
2) The supervisor LLM proposes a plan (surface, adsorbate, sites).
3) User clicks "Run Plan" to trigger structure building and energy
   calculations, then a short report is generated.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import dash
import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, State, dcc, html, no_update
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

from real_tools import real_energy_calculator, real_structure_builder


BASE_DIR = Path(__file__).resolve().parent


def load_avatar(file_name: str) -> Optional[str]:
    """Load an avatar and return a data URI, or None if unavailable."""

    path = BASE_DIR / file_name
    if not path.exists():
        return None

    ext = path.suffix.lower().lstrip(".")
    mime = "image/png" if ext == "png" else "image/jpeg"
    try:
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    except Exception:
        return None
    return f"data:{mime};base64,{encoded}"


AVATARS: Dict[str, Optional[str]] = {
    "assistant": load_avatar("agent.png"),
    "user": load_avatar("user.png"),
}


DEFAULT_MESSAGES: List[Dict[str, str]] = [
    {
        "role": "assistant",
        "content": (
            "Hello! I am DISCO-Pilot, your computational chemistry assistant. "
            "What system would you like to study?\n\n"
            "Example: Study the adsorption configuration of O atom on Pt(111) surface"
        ),
    }
]

# Default port for local run (can be overridden by PORT env)
DEFAULT_PORT = int(os.getenv("PORT", "8054"))


def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="deepseek-chat",
        openai_api_key=os.environ.get("DEEPSEEK_API_KEY"),
        openai_api_base="https://api.deepseek.com",
        temperature=0.1,
    )


def run_supervisor_planning(user_request: str) -> Dict[str, Any]:
    llm = get_llm()
    prompt = f"""
    You are a computational chemistry research supervisor. The user's request is: "{user_request}".

    Please extract the Surface and Adsorbate the user wants to study.
    Then, based on chemical knowledge, list the adsorption Sites to be tested on that surface.
    For fcc(111) surface, we focus on stable sites: top, fcc (hollow), hcp (hollow).
    (Note: Bridge sites often relax to hollow sites, so we skip them for efficiency).

    Please return strictly in JSON format, do not include Markdown formatting markers, format as follows:
    {{
        "surface": "Pt(111)",
        "adsorbate": "O",
        "sites": ["top", "fcc", "hcp"]
    }}
    """
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()
    if content.startswith("```json"):
        content = content[7:-3]
    elif content.startswith("```"):
        content = content[3:-3]
    return json.loads(content)


def render_chat(messages: List[Dict[str, str]]) -> List[html.Div]:
    items: List[html.Div] = []
    for msg in messages or []:
        role = msg.get("role") or "assistant"
        is_user = role == "user"
        avatar_src = AVATARS.get(role)
        bubble_class = "bubble user" if is_user else "bubble assistant"
        row_class = f"chat-row {'user' if is_user else 'assistant'}"
        sender_label = "You" if is_user else "DISCO-Pilot"

        items.append(
            html.Div(
                [
                    html.Div(
                        html.Img(
                            src=avatar_src,
                            alt=f"{role} avatar",
                            className="avatar",
                        )
                        if avatar_src
                        else html.Div(className="avatar avatar-fallback"),
                        className="avatar-wrap",
                    ),
                    html.Div(
                        [
                            html.Div(sender_label, className="msg-meta-name"),
                            dcc.Markdown(msg.get("content", ""), className="mb-0"),
                        ],
                        className=bubble_class,
                    ),
                ],
                className=row_class,
            )
        )
    return items or [html.Div("No messages yet", className="text-muted")]


def render_results_table(results: List[Dict[str, Any]]) -> html.Div:
    if not results:
        return html.Div("No calculations yet", className="text-muted")

    rows = []
    for item in results:
        val = f"{item['energy']:.4f} eV" if item.get("energy") is not None else "Failed"
        rows.append(
            html.Tr([
                html.Td(item.get("site", "")),
                html.Td(val),
                html.Td(os.path.basename(item.get("path", "")) if item.get("path") else "-"),
            ])
        )

    table = dbc.Table(
        [
            html.Thead(html.Tr([html.Th("Site"), html.Th("Energy"), html.Th("File")])),
            html.Tbody(rows),
        ],
        bordered=True,
        hover=True,
        size="sm",
        className="mb-0",
    )
    return table


app: Dash = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.10.5/font/bootstrap-icons.css"],
)
server = app.server


app.layout = dbc.Container(
    [
        dcc.Store(id="store-messages", data=DEFAULT_MESSAGES),
        dcc.Store(id="store-plan", data=None),
        dcc.Store(id="store-user-request", data=""),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H2("DISCO-Pilot Chat", className="mt-3 mb-1 fw-semibold"),
                        html.P(
                            "A focused, ChatGPT-style workspace for DISCO agent runs.",
                            className="text-muted mb-3",
                        ),
                    ],
                    width=8,
                ),
                dbc.Col(
                    dbc.Alert(
                        id="alert", is_open=False, dismissable=True, color="info", className="mt-3 mb-1"
                    ),
                    width=4,
                ),
            ],
            align="center",
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                html.Div(
                                    id="chat-thread",
                                    className="chat-scroller",
                                    style={"height": "560px", "overflowY": "auto"},
                                )
                            ),
                            className="mb-3 app-card chat-window",
                        ),
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    dbc.Textarea(
                                        id="prompt-input",
                                        placeholder="Type your request as you would in ChatGPT...",
                                        style={"height": "100px"},
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.Button(
                                                    "Generate Plan",
                                                    id="btn-plan",
                                                    color="primary",
                                                    className="w-100 mt-2",
                                                ),
                                                width=6,
                                            ),
                                            dbc.Col(
                                                dbc.ButtonGroup(
                                                    [
                                                        dbc.Button(
                                                            "Run Plan",
                                                            id="btn-run",
                                                            color="success",
                                                            disabled=True,
                                                        ),
                                                        dbc.Button(
                                                            "Cancel",
                                                            id="btn-cancel",
                                                            color="secondary",
                                                            outline=True,
                                                        ),
                                                    ],
                                                    className="w-100 mt-2",
                                                ),
                                                width=6,
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                            className="app-card composer-card",
                        ),
                    ],
                    width=8,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("API Key", className="py-2"),
                                dbc.CardBody(
                                    [
                                        dbc.Input(
                                            id="input-api-key",
                                            type="password",
                                            placeholder="DeepSeek API Key",
                                            value=os.getenv("DEEPSEEK_API_KEY", ""),
                                        ),
                                        html.Small(
                                            "Key is stored in-memory for this process only.",
                                            className="text-muted",
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-3 app-card",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Current Plan", className="py-2"),
                                dbc.CardBody(html.Div(id="plan-summary")),
                            ],
                            className="mb-3 app-card",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Execution Log", className="py-2"),
                                dbc.CardBody(
                                    html.Pre(
                                        id="execution-log",
                                        style={"height": "220px", "overflowY": "auto"},
                                        className="mb-0",
                                    )
                                ),
                            ],
                            className="mb-3 app-card",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Results", className="py-2"),
                                dbc.CardBody(html.Div(id="results-table")),
                            ],
                            className="app-card",
                        ),
                    ],
                    width=4,
                ),
            ]
        ),
    ],
    fluid=True,
    className="app-shell",
)


@app.callback(Output("chat-thread", "children"), Input("store-messages", "data"))
def on_messages_change(messages):
    return render_chat(messages)


@app.callback(
    Output("plan-summary", "children"),
    Output("btn-run", "disabled"),
    Input("store-plan", "data"),
    State("store-user-request", "data"),
)
def on_plan_change(plan, user_request):
    if not plan:
        return html.Div("No pending plan", className="text-muted"), True

    summary = html.Div(
        [
            html.Div([html.Strong("Surface:"), html.Span(f" {plan.get('surface', '-')}")]),
            html.Div([html.Strong("Adsorbate:"), html.Span(f" {plan.get('adsorbate', '-')}")]),
            html.Div([html.Strong("Sites:"), html.Span(" " + ", ".join(plan.get("sites", [])))]),
            html.Div([html.Strong("Task:"), html.Span(f" {user_request}")], className="mt-1"),
        ]
    )
    return summary, False


@app.callback(
    Output("store-messages", "data"),
    Output("store-plan", "data"),
    Output("store-user-request", "data"),
    Output("alert", "children"),
    Output("alert", "color"),
    Output("alert", "is_open"),
    Output("prompt-input", "value"),
    Input("btn-plan", "n_clicks"),
    State("prompt-input", "value"),
    State("store-messages", "data"),
    State("input-api-key", "value"),
    prevent_initial_call=True,
)
def on_generate_plan(n_clicks, prompt, messages, api_key):
    msgs = list(messages or [])
    if not prompt:
        return msgs, no_update, "", "Please enter a request first", "warning", True, no_update

    msgs.append({"role": "user", "content": prompt})

    if not api_key:
        return msgs, no_update, "", "Please set API key first", "danger", True, no_update

    os.environ["DEEPSEEK_API_KEY"] = api_key

    try:
        plan = run_supervisor_planning(prompt)
    except Exception as exc:  # pragma: no cover - defensive
        return msgs, None, "", f"Planning failed: {exc}", "danger", True, prompt

    response_md = (
        "Received. Proposed plan:\n\n"
        f"- Surface Model: `{plan['surface']}`\n"
        f"- Adsorbate: `{plan['adsorbate']}`\n"
        f"- Sites to Calculate: `{', '.join(plan['sites'])}`\n\n"
        "Click \"Run Plan\" to start execution."
    )
    msgs.append({"role": "assistant", "content": response_md})

    return msgs, plan, prompt, "Plan prepared", "success", True, ""


@app.callback(
    Output("store-messages", "data"),
    Output("store-plan", "data"),
    Output("store-user-request", "data"),
    Output("alert", "children"),
    Output("alert", "color"),
    Output("alert", "is_open"),
    Output("execution-log", "children"),
    Output("results-table", "children"),
    Input("btn-run", "n_clicks"),
    State("store-plan", "data"),
    State("store-user-request", "data"),
    State("store-messages", "data"),
    State("input-api-key", "value"),
    prevent_initial_call=True,
)
def on_run_plan(n_clicks, plan, user_request, messages, api_key):
    if not plan:
        return no_update, no_update, no_update, "No plan to run", "warning", True, no_update, no_update

    msgs = list(messages or [])

    if not api_key:
        return msgs, plan, user_request, "Please set API key first", "danger", True, no_update, no_update

    os.environ["DEEPSEEK_API_KEY"] = api_key

    logs: List[str] = []
    results: List[Dict[str, Any]] = []

    try:
        for site in plan.get("sites", []):
            logs.append(f"Building structure for site {site}...")
            path = real_structure_builder(plan.get("surface"), plan.get("adsorbate"), site)
            if path:
                logs.append(f"Created structure: {os.path.basename(path)}")
                energy = real_energy_calculator(path)
                if energy is not None:
                    logs.append(f"Energy for {site}: {energy:.4f} eV")
                    results.append({"site": site, "energy": energy, "path": path})
                else:
                    logs.append(f"Energy calculation failed for {site}")
                    results.append({"site": site, "energy": None, "path": path})
            else:
                logs.append(f"Structure build failed for {site}")
                results.append({"site": site, "energy": None, "path": None})

        report_text = generate_report(user_request, results)
        final_response = (
            "### Research Report\n\n"
            f"{report_text}\n\n"
            "Data Summary:\n" + "\n".join(
                [
                    f"- {item['site']}: {item['energy']:.4f} eV" if item.get("energy") is not None else f"- {item['site']}: Failed"
                    for item in results
                ]
            )
        )
        msgs.append({"role": "assistant", "content": final_response})

    except Exception as exc:  # pragma: no cover - defensive
        logs.append(f"Execution failed: {exc}")
        return msgs, plan, user_request, f"Execution failed: {exc}", "danger", True, "\n".join(logs), no_update

    return (
        msgs,
        None,
        "",
        "Execution finished",
        "success",
        True,
        "\n".join(logs),
        render_results_table(results),
    )


@app.callback(
    Output("store-plan", "data"),
    Output("store-user-request", "data"),
    Output("alert", "children"),
    Output("alert", "color"),
    Output("alert", "is_open"),
    Input("btn-cancel", "n_clicks"),
    prevent_initial_call=True,
)
def on_cancel(n_clicks):
    return None, "", "Plan cleared", "secondary", True


if __name__ == "__main__":
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        except Exception:
            pass
    # Disable reloader in Dash 3 to avoid parent exit + 502
    app.run(debug=False, port=DEFAULT_PORT, host="0.0.0.0", use_reloader=False)

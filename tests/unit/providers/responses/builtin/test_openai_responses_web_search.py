# Copyright (c) The OGX Contributors.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.

import pytest
from pydantic import TypeAdapter

from ogx_api import (
    InvalidParameterError,
    WebSearchActionFind,
    WebSearchActionOpenPage,
    WebSearchActionSearch,
    WebSearchSource,
)
from ogx_api.openai_responses import (
    OpenAIResponseInputToolWebSearch,
    OpenAIResponseOutput,
    WebSearchToolTypes,
)
from ogx_api.responses.models import CreateResponseRequest
from ogx_api.router_utils import try_translate_to_http_exception
from ogx_api.tools import ToolDef, ToolInvocationResult
from tests.unit.providers.responses.builtin.test_openai_responses_helpers import fake_stream


async def test_web_search_output_includes_action_with_sources(openai_responses_impl, mock_inference_api):
    """Test that web search output includes action with source URLs."""
    input_text = "Find papers on transformers"
    model = "meta-llama/Llama-3.1-8B-Instruct"

    openai_responses_impl.tool_groups_api.get_tool.return_value = ToolDef(
        name="web_search",
        toolgroup_id="web_search",
        description="Search the web for information",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The query to search for"}},
            "required": ["query"],
        },
    )

    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = ToolInvocationResult(
        status="completed",
        content="Attention Is All You Need",
        metadata={
            "query": "What is the capital of Ireland?",
            "sources": [
                {"url": "https://arxiv.org/abs/1706.03762"},
                {"url": "https://en.wikipedia.org/wiki/Transformer"},
            ],
        },
    )

    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("tool_call_completion.yaml"),
        fake_stream(),
    ]

    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input=input_text, model=model, temperature=0.1, tools=[OpenAIResponseInputToolWebSearch(type="web_search")]
        )
    )

    # Find the web search tool call in output
    web_search_output = next(
        (o for o in result.output if o.type == "web_search_call"),
        None,
    )
    assert web_search_output is not None
    assert web_search_output.action is not None
    assert web_search_output.action.type == "search"
    assert len(web_search_output.action.sources) == 2
    assert web_search_output.action.sources[0].url == "https://arxiv.org/abs/1706.03762"


def _setup_web_search_mocks(openai_responses_impl, mock_inference_api):
    """Set up common mocks for web search config tests."""
    openai_responses_impl.tool_groups_api.get_tool.return_value = ToolDef(
        name="web_search",
        toolgroup_id="web_search",
        description="Search the web",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    )
    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = ToolInvocationResult(
        status="completed",
        content="Dublin",
    )
    mock_inference_api.openai_chat_completion.side_effect = [
        fake_stream("tool_call_completion.yaml"),
        fake_stream(),
    ]


_OPEN_PAGE = WebSearchActionOpenPage(url="https://example.com/a")
_FIND = WebSearchActionFind(url="https://example.com/a", pattern="needle")


def _web_search_result(metadata=None, error_code=None):
    return ToolInvocationResult(status="completed", content="Dublin", metadata=metadata, error_code=error_code)


async def _run_web_search(openai_responses_impl, stream):
    result = await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What is the capital of Ireland?",
            model="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.1,
            tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
            stream=stream,
        )
    )
    if not stream:
        return next(o for o in result.output if o.type == "web_search_call")
    chunks = [c async for c in result]
    done = [c.item for c in chunks if c.type == "response.output_item.done" and c.item.type == "web_search_call"]
    assert len(done) == 1
    return done[0]


@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
@pytest.mark.parametrize(
    "action, expected_type",
    [(_OPEN_PAGE, "open_page"), (_FIND, "find_in_page")],
    ids=["open_page", "find_in_page"],
)
async def test_web_search_reports_typed_action_from_provider(
    openai_responses_impl, mock_inference_api, action, expected_type, stream
):
    _setup_web_search_mocks(openai_responses_impl, mock_inference_api)
    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = _web_search_result({"action": action})

    item = await _run_web_search(openai_responses_impl, stream)

    assert item.status == "completed"
    assert item.action.type == expected_type
    assert item.action.url == "https://example.com/a"
    if expected_type == "find_in_page":
        assert item.action.pattern == "needle"


_SOURCES = [{"url": "https://example.com/s"}, {"title": "no url"}, {"url": "https://example.com/t"}]
_SEARCH_SOURCES = [WebSearchSource(url="https://example.com/s"), WebSearchSource(url="https://example.com/t")]
# The model's tool-call query from the mocked inference fixture; used when metadata has none.
_TOOL_CALL_QUERY = "What is the capital of Ireland?"


@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
@pytest.mark.parametrize(
    "metadata, expected",
    [
        (
            {"query": "q", "sources": _SOURCES},
            WebSearchActionSearch(query="q", queries=["q"], sources=_SEARCH_SOURCES),
        ),
        (
            {"query": "q", "sources": _SOURCES, "action": None},
            WebSearchActionSearch(query="q", queries=["q"], sources=_SEARCH_SOURCES),
        ),
        (
            {"query": "q", "sources": _SOURCES, "action": {"type": "open_page", "url": "https://example.com/a"}},
            WebSearchActionSearch(query="q", queries=["q"], sources=_SEARCH_SOURCES),
        ),
        (
            {
                "query": "q",
                "sources": _SOURCES,
                "action": WebSearchActionSearch(query="other", queries=["other"]),
            },
            WebSearchActionSearch(query="q", queries=["q"], sources=_SEARCH_SOURCES),
        ),
        (
            {"sources": _SOURCES},
            WebSearchActionSearch(query=_TOOL_CALL_QUERY, queries=[_TOOL_CALL_QUERY], sources=_SEARCH_SOURCES),
        ),
    ],
    ids=["no_action", "action_none", "dict_action", "search_object", "no_query_in_metadata"],
)
async def test_web_search_without_usable_typed_action_stays_a_search(
    openai_responses_impl, mock_inference_api, metadata, expected, stream
):
    _setup_web_search_mocks(openai_responses_impl, mock_inference_api)
    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = _web_search_result(metadata)

    item = await _run_web_search(openai_responses_impl, stream)

    assert item.status == "completed"
    assert item.action == expected


@pytest.mark.parametrize("action", [_OPEN_PAGE, _FIND], ids=["open_page", "find_in_page"])
async def test_web_search_typed_action_survives_json_round_trip(openai_responses_impl, mock_inference_api, action):
    _setup_web_search_mocks(openai_responses_impl, mock_inference_api)
    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = _web_search_result({"action": action})

    item = await _run_web_search(openai_responses_impl, stream=False)
    restored = TypeAdapter(OpenAIResponseOutput).validate_json(item.model_dump_json())

    assert type(restored.action) is type(action)
    assert restored.action == action


@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
async def test_web_search_failure_has_no_action_even_with_typed_action(
    openai_responses_impl, mock_inference_api, stream
):
    _setup_web_search_mocks(openai_responses_impl, mock_inference_api)
    openai_responses_impl.tool_runtime_api.invoke_tool.return_value = _web_search_result(
        {"action": _OPEN_PAGE}, error_code=1
    )

    item = await _run_web_search(openai_responses_impl, stream)

    assert item.status == "failed"
    assert item.action is None


async def test_web_search_with_filters_and_location(openai_responses_impl, mock_inference_api):
    """Test that web search filters and user_location are passed to invoke_tool."""
    _setup_web_search_mocks(openai_responses_impl, mock_inference_api)
    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What is the capital of Ireland?",
            model="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.1,
            tools=[
                OpenAIResponseInputToolWebSearch(
                    type="web_search",
                    search_context_size="high",
                    filters={"allowed_domains": ["arxiv.org", "scholar.google.com"]},
                    user_location={"type": "approximate", "country": "US", "city": "San Francisco"},
                )
            ],
        )
    )
    call_kwargs = openai_responses_impl.tool_runtime_api.invoke_tool.call_args.kwargs["kwargs"]
    assert call_kwargs["query"] == "What is the capital of Ireland?"
    assert call_kwargs["allowed_domains"] == ["arxiv.org", "scholar.google.com"]
    assert call_kwargs["user_location"]["country"] == "US"
    assert call_kwargs["user_location"]["city"] == "San Francisco"
    assert call_kwargs["search_context_size"] == "high"


async def test_web_search_without_config_passes_only_query(openai_responses_impl, mock_inference_api):
    """Test that web search without filters/location only passes query."""
    _setup_web_search_mocks(openai_responses_impl, mock_inference_api)
    await openai_responses_impl.create_openai_response(
        CreateResponseRequest(
            input="What is the capital of Ireland?",
            model="meta-llama/Llama-3.1-8B-Instruct",
            temperature=0.1,
            tools=[OpenAIResponseInputToolWebSearch(type="web_search")],
        )
    )
    call_kwargs = openai_responses_impl.tool_runtime_api.invoke_tool.call_args.kwargs["kwargs"]
    assert call_kwargs["query"] == "What is the capital of Ireland?"
    assert "search_context_size" not in call_kwargs
    assert "allowed_domains" not in call_kwargs
    assert "user_location" not in call_kwargs


@pytest.mark.parametrize("stream", [True, False], ids=["streaming", "non_streaming"])
@pytest.mark.parametrize("tool_type", WebSearchToolTypes)
async def test_unregistered_web_search_tool_is_rejected_with_400_before_the_response_starts(
    openai_responses_impl, mock_inference_api, mock_tool_groups_api, tool_type, stream
):
    mock_tool_groups_api.get_tool.side_effect = ValueError("Tool 'web_search' not found")

    with pytest.raises(InvalidParameterError, match="web_search") as exc_info:
        await openai_responses_impl.create_openai_response(
            CreateResponseRequest(
                input="What is the capital of Ireland?",
                model="meta-llama/Llama-3.1-8B-Instruct",
                stream=stream,
                store=False,
                tools=[OpenAIResponseInputToolWebSearch(type=tool_type)],
            )
        )

    http_exc = try_translate_to_http_exception(exc_info.value)
    assert http_exc is not None
    assert http_exc.status_code == 400
    mock_inference_api.openai_chat_completion.assert_not_called()

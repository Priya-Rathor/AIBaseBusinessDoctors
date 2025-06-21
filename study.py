from typing import TypedDict, Annotated, Optional, List, Dict, Any, Union
from langgraph.graph import add_messages, StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from fastapi import FastAPI, Query, HTTPException, status, Path
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
import json
from uuid import uuid4
from langgraph.checkpoint.memory import MemorySaver
import logging
from pydantic import BaseModel, Field
import re

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize memory saver for checkpointing
memory = MemorySaver()

# Define types
class State(TypedDict):
    messages: Annotated[List[Union[HumanMessage, ToolMessage, AIMessageChunk]], add_messages]

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000)
    checkpoint_id: Optional[str] = Field(None, pattern=r'^[a-f0-9-]{36}$')

class ToolCall(BaseModel):
    name: str
    args: Dict[str, Any]
    id: str

# Initialize tools
search_tool = TavilySearchResults(max_results=4)
tools = [search_tool]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", streaming=True, temperature=0.7)
llm_with_tools = llm.bind_tools(tools=tools)

# Helper functions
def validate_message_content(message: str) -> str:
    message = message.strip()
    if not message:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message cannot be empty")
    message = re.sub(r'<script.*?>.*?</script>', '', message, flags=re.IGNORECASE)
    return message[:1000]

def serialize_ai_message_chunk(chunk):
    if not isinstance(chunk, AIMessageChunk):
        return {"type": "error", "content": "Invalid chunk type"}
    return {"type": "content", "content": chunk.content}

# Graph nodes
async def model(state: State):
    try:
        result = await llm_with_tools.ainvoke(state["messages"])
        logger.info("Model invocation successful")
        return {"messages": [result]}
    except Exception as e:
        logger.error(f"Model error: {e}")
        raise HTTPException(status_code=503, detail="Model service unavailable")

async def tools_router(state: State):
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END

async def tool_node(state: State):
    tool_calls = state["messages"][-1].tool_calls
    assistant_message = state["messages"][-1]
    tool_messages = [assistant_message]

    for tool_call in tool_calls:
        if isinstance(tool_call, ToolCall):
            tool_call = tool_call.dict()
        try:
            if tool_call["name"] == "tavily_search_results_json":
                raw_results = await search_tool.ainvoke(tool_call["args"])
                urls = [r["url"] for r in raw_results if "url" in r]
                result_msg = {
                    "type": "search_results",
                    "data": urls
                }
                tool_messages.append(ToolMessage(
                    content=json.dumps(result_msg),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                ))
        except Exception as e:
            logger.error(f"Tool error: {e}")
            tool_messages.append(ToolMessage(
                content=json.dumps({"error": str(e)}),
                tool_call_id=tool_call["id"],
                name=tool_call["name"]
            ))
    return {"messages": tool_messages}

# Graph
graph_builder = StateGraph(State)
graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")
graph_builder.add_conditional_edges("model", tools_router)
graph_builder.add_edge("tool_node", "model")
graph = graph_builder.compile(checkpointer=memory)

# FastAPI app
app = FastAPI(title="Chat Stream API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/chat_stream")
async def chat_stream(message: str = Query(...), checkpoint_id: Optional[str] = Query(None)):
    message = validate_message_content(message)

    async def generate():
        is_new = checkpoint_id is None
        new_id = str(uuid4()) if is_new else checkpoint_id
        config = {"configurable": {"thread_id": new_id}}

        if is_new:
            yield f"data: {json.dumps({'type': 'checkpoint', 'checkpoint_id': new_id})}\n\n"

        initial_state = {"messages": [HumanMessage(content=message)]}

        if not is_new:
            checkpoint = await memory.aget(config)
            if checkpoint and "messages" in checkpoint:
                logger.debug(f"Replaying {len(checkpoint['messages'])} messages")
                initial_state["messages"] = checkpoint["messages"] + initial_state["messages"]

        async for event in graph.astream_events(initial_state, version="v2", config=config):
            try:
                # Ensure event is a dictionary before processing
                if not isinstance(event, dict):
                    logger.error(f"Unexpected event format: {event}")
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Unexpected event format'})}\n\n"
                    continue

                event_type = event.get("event")
                if not event_type:
                    continue

                # Safely access nested dictionary keys
                data = event.get("data", {})
                chunk = data.get("chunk") if isinstance(data, dict) else None

                if event_type == "on_chat_model_stream":
                    if isinstance(chunk, AIMessageChunk):
                        yield f"data: {json.dumps(serialize_ai_message_chunk(chunk))}\n\n"

                elif event_type in ["on_chat_model_end", "on_llm_end", "on_chain_end"]:
                    output = data.get("output", {}) if isinstance(data, dict) else {}
                    messages = output.get("messages", []) if isinstance(output, dict) else []
                    
                    for msg in messages:
                        if hasattr(msg, "content") and msg.content:
                            try:
                                content_obj = json.loads(msg.content)
                                if isinstance(content_obj, dict) and "type" in content_obj:
                                    yield f"data: {json.dumps(content_obj)}\n\n"
                                else:
                                    yield f"data: {json.dumps({'type': 'content', 'content': msg.content})}\n\n"
                            except json.JSONDecodeError:
                                yield f"data: {json.dumps({'type': 'content', 'content': msg.content})}\n\n"
                        if hasattr(msg, "tool_calls"):
                            for tool_call in msg.tool_calls:
                                args = getattr(tool_call, "args", {})
                                query = args.get("query", "unknown") if isinstance(args, dict) else "unknown"
                                yield f"data: {json.dumps({'type': 'search_start', 'query': query})}\n\n"

                elif event_type == "on_tool_end" and event.get("name") == "tavily_search_results_json":
                    output = data.get("output") if isinstance(data, dict) else None
                    try:
                        if output:
                            tool_content = json.loads(output)
                            yield f"data: {json.dumps(tool_content)}\n\n"
                    except Exception:
                        yield f"data: {json.dumps({'type': 'search_error', 'message': 'Failed to process search results'})}\n\n"

                elif event_type == "on_tool_error" and event.get("name") == "tavily_search_results_json":
                    error = data.get("error", "Search failed") if isinstance(data, dict) else "Search failed"
                    yield f"data: {json.dumps({'type': 'search_error', 'message': error})}\n\n"

            except Exception as e:
                logger.error(f"Stream error: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'end'})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/conversations/{checkpoint_id}")
async def get_conversation(checkpoint_id: str = Path(..., regex=r'^[a-f0-9-]{36}$')):
    checkpoint = await memory.aget({"configurable": {"thread_id": checkpoint_id}})
    if not checkpoint:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return jsonable_encoder(checkpoint)
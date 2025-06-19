from typing import TypedDict, Annotated, Optional, List, Dict, Any
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
    messages: Annotated[List, add_messages]

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
    """Validate and sanitize user message."""
    message = message.strip()
    if not message:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Message cannot be empty"
        )
    # Basic XSS protection
    message = re.sub(r'<script.*?>.*?</script>', '', message, flags=re.IGNORECASE)
    return message[:1000]  # Limit length

def process_search_results(raw_results: Any) -> List[Dict[str, str]]:
    """Process raw search results into a standardized format."""
    search_data = []
    
    if isinstance(raw_results, str):
        try:
            raw_results = json.loads(raw_results)
        except json.JSONDecodeError:
            return [{"url": "", "title": "Raw content", "content": raw_results[:200] + "..."}]
    
    if isinstance(raw_results, dict):
        # Handle direct result
        if 'url' in raw_results or 'link' in raw_results:
            search_data.append({
                "url": raw_results.get('url') or raw_results.get('link', ''),
                "title": raw_results.get('title', 'No title available'),
                "content": (raw_results.get('content', '')[:200] + '...') if raw_results.get('content') else 'No content available'
            })
        # Handle results in 'results' field
        elif 'results' in raw_results:
            for result in raw_results['results']:
                if isinstance(result, dict):
                    search_data.append({
                        "url": result.get('url') or result.get('link', ''),
                        "title": result.get('title', 'No title available'),
                        "content": (result.get('content', '')[:200] + '...') if result.get('content') else 'No content available'
                    })
    
    elif isinstance(raw_results, list):
        for item in raw_results:
            if isinstance(item, dict):
                search_data.append({
                    "url": item.get('url') or item.get('link', ''),
                    "title": item.get('title', 'No title available'),
                    "content": (item.get('content', '')[:200] + '...') if item.get('content') else 'No content available'
                })
    
    return search_data

# Graph nodes and edges
async def model(state: State):
    """Process messages with the LLM."""
    try:
        result = await llm_with_tools.ainvoke(state["messages"])
        logger.info("Model invocation successful")
        return {"messages": [result]}
    except Exception as e:
        logger.error(f"Model invocation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model service unavailable"
        )

async def tools_router(state: State):
    """Route to tool node if tool calls are present."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tool_node"
    return END

async def tool_node(state: State):
    """Handle tool calls from the LLM."""
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []
    
    # Keep the assistant message that contains the tool calls
    assistant_message = state["messages"][-1]
    tool_messages.append(assistant_message)
    
    for tool_call in tool_calls:
        try:
            if tool_call["name"] == "tavily_search_results_json":
                search_results = await search_tool.ainvoke(tool_call["args"])
                
                # Create a special search results message
                urls = [result["url"] for result in search_results if "url" in result]
                search_results_message = {
                    "type": "search_results",
                    "search_results": urls
                }
                tool_message = ToolMessage(
                    content=json.dumps(search_results_message),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
                tool_messages.append(tool_message)
                
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            tool_message = ToolMessage(
                content=json.dumps({"error": str(e)}),
                tool_call_id=tool_call["id"],
                name=tool_call["name"]
            )
            tool_messages.append(tool_message)
    
    return {"messages": tool_messages}


# Build the graph
graph_builder = StateGraph(State)
graph_builder.add_node("model", model)
graph_builder.add_node("tool_node", tool_node)
graph_builder.set_entry_point("model")
graph_builder.add_conditional_edges("model", tools_router)
graph_builder.add_edge("tool_node", "model")
graph = graph_builder.compile(checkpointer=memory)

# Initialize FastAPI
app = FastAPI(
    title="Chat Stream API",
    description="API for streaming chat responses with memory",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def serialize_ai_message_chunk(chunk):
    """Serialize AI message chunks for streaming."""
    if isinstance(chunk, AIMessageChunk):
        return chunk.content
    raise TypeError(f"Object of type {type(chunk).__name__} is not correctly formatted for serialization")

@app.get("/chat_stream")
async def chat_stream(
    message: str = Query(..., min_length=1, max_length=1000),
    checkpoint_id: Optional[str] = Query(None, regex=r'^[a-f0-9-]{36}$')
):
    """Stream chat responses with optional conversation memory."""
    try:
        message = validate_message_content(message)
        
        async def generate_responses():
            is_new_conversation = checkpoint_id is None
            new_checkpoint_id = str(uuid4()) if is_new_conversation else checkpoint_id
            
            config = {
                "configurable": {
                    "thread_id": new_checkpoint_id
                }
            }

            if is_new_conversation:
                yield f"data: {json.dumps({'type': 'checkpoint', 'checkpoint_id': new_checkpoint_id})}\n\n"

            try:
                # Initialize state with previous messages if continuing conversation
                initial_state = {"messages": [HumanMessage(content=message)]}
                
                if not is_new_conversation:
                    checkpoint = await memory.aget(config)
                    if checkpoint and isinstance(checkpoint, dict) and "messages" in checkpoint:
                        initial_state["messages"] = checkpoint["messages"] + initial_state["messages"]

                async for event in graph.astream_events(
                    initial_state,
                    version="v2",
                    config=config
                ):
                    try:
                        event_type = event.get("event")
                        if not event_type:
                            continue

                        # Handle content events
                        if event_type == "on_chat_model_stream":
                            if isinstance(event.get("data", {}).get("chunk"), AIMessageChunk):
                                chunk = event["data"]["chunk"]
                                content = serialize_ai_message_chunk(chunk)
                                yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"

                        elif event_type in ["on_chat_model_end", "on_llm_end", "on_chain_end"]:
                            output = event.get("data", {}).get("output", {})
                            messages = output.get("messages", []) if isinstance(output, dict) else []
                            
                            for msg in messages:
                                if hasattr(msg, "content") and msg.content:
                                    yield f"data: {json.dumps({'type': 'content', 'content': msg.content})}\n\n"
                                if hasattr(msg, "tool_calls"):
                                    for tool_call in msg.tool_calls:
                                        if isinstance(tool_call, dict) and tool_call.get("name") == "tavily_search_results_json":
                                            yield f"data: {json.dumps({'type': 'search_start', 'query': tool_call.get('args', {}).get('query', '')})}\n\n"

                        # Handle search results
                        elif event_type == "on_tool_end" and event.get("name") == "tavily_search_results_json":
                            output = event.get("data", {}).get("output")
                            
                            try:
                                if output:
                                    tool_content = json.loads(output)
                                    # First send processed search results
                                    if "type" in tool_content and tool_content["type"] == "search_results":
                                         yield f"data: {json.dumps(tool_content)}\n\n"
                                
                                else:
                                    yield f"data: {json.dumps({'type': 'search_error', 'message': 'Empty search results'})}\n\n"
                            except Exception as e:
                                logger.error(f"Error processing search results: {str(e)}\nRaw output: {output}")
                                yield f"data: {json.dumps({'type': 'search_error', 'message': 'Failed to process search results'})}\n\n"

                        elif event_type == "on_tool_error" and event.get("name") == "tavily_search_results_json":
                            error = event.get("data", {}).get("error", "Search failed")
                            yield f"data: {json.dumps({'type': 'search_error', 'message': str(error)})}\n\n"

                    except Exception as e:
                        logger.error(f"Error processing event: {str(e)}", exc_info=True)
                        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                        continue

            except Exception as e:
                logger.error(f"Stream error: {str(e)}", exc_info=True)
                yield f"data: {json.dumps({'type': 'error', 'message': 'Stream processing failed'})}\n\n"
                return

            yield f"data: {json.dumps({'type': 'end'})}\n\n"

        return StreamingResponse(
            generate_responses(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )

    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

@app.get("/conversations/{checkpoint_id}")
async def get_conversation(checkpoint_id: str = Path(..., regex=r'^[a-f0-9-]{36}$')):
    """Get conversation history from checkpoint."""
    try:
        checkpoint = await memory.aget({"configurable": {"thread_id": checkpoint_id}})
        if not checkpoint:
            raise HTTPException(status_code=404, detail="Conversation not found")
        return jsonable_encoder(checkpoint)
    except Exception as e:
        logger.error(f"Failed to get conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
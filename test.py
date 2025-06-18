from typing import TypedDict, Annotated, Optional, List, Dict, Any
from langgraph.graph import add_messages, StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessageChunk, ToolMessage
from dotenv import load_dotenv
from langchain_community.tools.tavily_search import TavilySearchResults
from fastapi import FastAPI, Query, HTTPException, status
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
def sanitize_content(content: str) -> str:
    """Sanitize content for SSE streaming."""
    return json.dumps({"content": content})[1:-1]  # Remove outer quotes from JSON string

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
    
    for tool_call in tool_calls:
        try:
            if tool_call["name"] == "tavily_search_results_json":
                search_results = await search_tool.ainvoke(tool_call["args"])
                tool_message = ToolMessage(
                    content=str(search_results),
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"]
                )
                tool_messages.append(tool_message)
        except Exception as e:
            logger.error(f"Tool execution failed: {str(e)}")
            tool_message = ToolMessage(
                content=f"Error executing tool: {str(e)}",
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

# SSE endpoint
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
                yield f"event: checkpoint\ndata: {json.dumps({'checkpoint_id': new_checkpoint_id})}\n\n"

            try:
                # Initialize with previous messages if continuing conversation
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
                    event_type = event.get("event")
                    
                    # Skip if event type is missing
                    if not event_type:
                        continue
                        
                    # Handle different event types safely
                    if event_type == "on_chat_model_stream":
                        chunk = event.get("data", {}).get("chunk")
                        if isinstance(chunk, AIMessageChunk):
                            yield f"event: content\ndata: {json.dumps({'content': chunk.content})}\n\n"

                    elif event_type in ["on_chat_model_end", "on_llm_end", "on_chain_end"]:
                        output = event.get("data", {}).get("output")
                        
                        # Handle case where output might be a string or dict
                        messages = []
                        if isinstance(output, dict):
                            messages = output.get("messages", [])
                        elif hasattr(output, "messages"):  # Handle case where output is an object
                            messages = getattr(output, "messages", [])
                        
                        for msg in messages:
                            if hasattr(msg, "content"):
                                yield f"event: content\ndata: {json.dumps({'content': msg.content})}\n\n"
                            if hasattr(msg, "tool_calls"):
                                for tool_call in msg.tool_calls:
                                    if tool_call.get("name") == "tavily_search_results_json":
                                        yield f"event: search_start\ndata: {json.dumps({'query': tool_call.get('args', {}).get('query', '')})}\n\n"

                    elif event_type == "on_tool_end" and event.get("name") == "tavily_search_results_json":
                        output = event.get("data", {}).get("output")
                        if isinstance(output, list):
                            urls = [item.get("url") for item in output if isinstance(item, dict) and item.get("url")]
                            urls = [url for url in urls if url]  # Filter out None values
                            if urls:
                                yield f"event: search_results\ndata: {json.dumps({'urls': urls})}\n\n"

            except Exception as e:
                logger.error(f"Stream error: {str(e)}", exc_info=True)
                yield f"event: error\ndata: {json.dumps({'message': 'An error occurred during streaming'})}\n\n"
                return

            yield "event: end\ndata: {}\n\n"

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

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

# Example of adding more endpoints
from fastapi import Path  # Add this import at the top

# Change this endpoint
@app.get("/conversations/{checkpoint_id}")
async def get_conversation(checkpoint_id: str = Path(..., regex=r'^[a-f0-9-]{36}$')):
    """Get conversation history from checkpoint."""
    try:
        checkpoint = await memory.aget({"configurable": {"thread_id": checkpoint_id}})
        return jsonable_encoder(checkpoint)
    except Exception as e:
        logger.error(f"Failed to get conversation: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
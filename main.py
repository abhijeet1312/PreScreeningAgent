"""
Main FastAPI Server
Handles Twilio integration and WebSocket communication
"""
import logging
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import Response
from twilio.rest import Client
from twilio.twiml.voice_response import VoiceResponse, Connect
from pydantic import BaseModel
from config import Config
from voice_agent import VoiceAgent

from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware

import socketio

# Setup Async Socket.io Server



# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Active call tracking
active_calls = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    # Startup
    try:
        Config.validate()
        logger.info("✅ Configuration validated")
    except ValueError as e:
        logger.error(f"❌ Configuration error: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down...")

class CallRequest(BaseModel):
    phone: str
    workflow_type: str
    workflow_data: dict = {}
    
class WorkflowUpdate(BaseModel):
    chat_id: str
    event: str = "function_status"
    data: dict
# Initialize FastAPI
app = FastAPI(
    title="Sarvam Voice Agent API",
    description="Real-time voice agent powered by Sarvam AI and Twilio",
    version="1.0.0",
    lifespan=lifespan
)



# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ✅ Allow all origins
    allow_credentials=False,  # ❌ Must be False when using allow_origins=["*"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3️⃣ Create Socket.IO server
sio = socketio.AsyncServer(
    async_mode="asgi",
    cors_allowed_origins="*"
)

# 4️⃣ Wrap FastAPI app
socket_app = socketio.ASGIApp(sio, app)

@sio.on("join_chat_room")
async def handle_join(sid, chat_id):
    await sio.enter_room(sid, chat_id)
    print(f"User {sid} joined room: {chat_id}")
    
@app.post("/api/workflow-update")
async def workflow_update(payload: WorkflowUpdate):
    await sio.emit(
        payload.event,
        payload.data,
        room=payload.chat_id
    )
    return {"status": "success", "message": "Notification broadcasted"}

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "Sarvam Voice Agent",
        "active_calls": len(active_calls),
        "version": "1.0.0",
        "provider": {
            "transcriber": "Sarvam AI STT",
            "synthesizer": "Sarvam AI TTS",
            "llm": "Azure OpenAI",
            "telephony": "Twilio"
        }
    }


@app.get("/health")
async def health():
    """Detailed health check"""
    return {
        "healthy": True,
        "active_calls": len(active_calls),
        "call_ids": list(active_calls.keys())
    }


@app.post("/voice/incoming")
async def incoming_call():
    """
    Handle incoming Twilio voice call
    Returns TwiML to connect call to WebSocket stream
    """
    logger.info("📞 Incoming call received")
    
    response = VoiceResponse()
    connect = Connect()
    
    # Build WebSocket URL
    base_url = Config.WEBHOOK_BASE_URL
    base_url = base_url.replace("https://", "").replace("http://", "")
    
    # Connect to WebSocket stream
    connect.stream(url=f"wss://{base_url}/stream")
    response.append(connect)
    
    return Response(content=str(response), media_type="application/xml")


@app.websocket("/stream")
async def stream_handler(websocket: WebSocket):
    """
    Main WebSocket handler for Twilio Media Streams
    Manages call lifecycle and agent orchestration
    """
    await websocket.accept()
    
    agent: VoiceAgent = None
    
    logger.info("🔌 WebSocket connected")
    
    try:
        async for message in websocket.iter_text():
            data = await asyncio.to_thread(lambda: __import__('json').loads(message))
            event = data.get("event")
            
            # Connection handshake
            if event == "connected":
                logger.info("🔗 Twilio Media Stream connected")
            
            # Call started
            elif event == "start":
                call_sid = data["start"]["callSid"]
                stream_sid = data["start"]["streamSid"]
                
                logger.info(f"📞 Call started: {call_sid}")
                
                # Create and initialize agent
                # agent = VoiceAgent(call_sid, stream_sid, websocket)
                # get workflow for this call
                workflow_info = active_calls.get(call_sid, {
                    "workflow_type": "default",
                    "workflow_data": {}
                })

                agent = VoiceAgent(
                    call_sid=call_sid,
                    stream_sid=stream_sid,
                    websocket=websocket,
                    workflow_type=workflow_info["workflow_type"],
                    workflow_data=workflow_info["workflow_data"],
                )

                active_calls[call_sid] = agent
                
                try:
                    # Initialize components
                    await agent.initialize()
                    
                    # Start conversation
                    await agent.start_conversation()
                    
                except Exception as e:
                    logger.error(f"❌ Agent initialization error: {e}")
                    if call_sid in active_calls:
                        del active_calls[call_sid]
                    raise
            
            # Audio data
            elif event == "media" and agent:
                payload = data["media"]["payload"]
                await agent.process_audio(payload)
            
            # Call stopped
            elif event == "stop":
                logger.info("🛑 Call ended by Twilio")
                
                if agent:
                    await agent.cleanup()
                    
                    if agent.call_sid in active_calls:
                        del active_calls[agent.call_sid]
            
            # Mark event (for debugging)
            elif event == "mark":
                logger.debug(f"🏷️ Mark event: {data.get('mark', {})}")
            
            # Other events
            else:
                logger.debug(f"📩 Unhandled event: {event}")
    
    except WebSocketDisconnect:
        logger.info("🔌 WebSocket disconnected")
        
        if agent:
            await agent.cleanup()
            
            if agent.call_sid in active_calls:
                del active_calls[agent.call_sid]
    
    except Exception as e:
        logger.error(f"❌ WebSocket error: {e}")
        import traceback
        traceback.print_exc()
        
        if agent:
            await agent.cleanup()
            
            if agent.call_sid in active_calls:
                del active_calls[agent.call_sid]


# @app.post("/api/call")
# async def make_outbound_call(request_data: dict):
#     """
#     Initiate outbound call
    
#     Request body:
#     {
#         "phone": "+1234567890",
#         "from": "+0987654321" (optional, uses default if not provided)
#     }
#     """
#     phone = request_data.get("phone")
#     from_number = request_data.get("from", Config.TWILIO_PHONE_NUMBER)
    
#     if not phone:
#         return {"success": False, "error": "Phone number is required"}
    
#     logger.info(f"📞 Initiating outbound call to {phone}")
    
#     try:
#         client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)
        
#         call = client.calls.create(
#             from_=from_number,
#             to=phone,
#             url=f"{Config.WEBHOOK_BASE_URL}/voice/incoming",
#             record=True,
#             recording_status_callback=f"{Config.WEBHOOK_BASE_URL}/api/recording",
#         )
        
#         logger.info(f"✅ Call initiated: {call.sid}")
        
#         return {
#             "success": True,
#             "call_sid": call.sid,
#             "to": phone,
#             "from": from_number,
#             "status": call.status
#         }
    
#     except Exception as e:
#         logger.error(f"❌ Outbound call error: {e}")
#         return {
#             "success": False,
#             "error": str(e)
#         }
@app.post("/api/call")
async def make_outbound_call(request_data: dict):
    phone = request_data.get("phone")
    workflow_type = request_data.get("workflow_type", "default")
    workflow_data = request_data.get("workflow_data", {})
    from_number = request_data.get("from", Config.TWILIO_PHONE_NUMBER)

    if not phone:
        return {"success": False, "error": "Phone number is required"}

    logger.info(f"📞 Initiating outbound call to {phone} (workflow: {workflow_type})")

    try:
        # Temporarily store workflow info before Twilio gives us call_sid
        active_calls[f"pending:{phone}"] = {
            "workflow_type": workflow_type,
            "workflow_data": workflow_data
        }

        client = Client(Config.TWILIO_ACCOUNT_SID, Config.TWILIO_AUTH_TOKEN)

        call = client.calls.create(
            from_=from_number,
            to=phone,
            url=f"{Config.WEBHOOK_BASE_URL}/voice/incoming",
            record=True,
            recording_status_callback=f"{Config.WEBHOOK_BASE_URL}/api/recording"
        )

        logger.info(f"✅ Call initiated (Twilio Call SID: {call.sid})")

        # Now store workflow info against the real call SID
        active_calls[call.sid] = {
            "workflow_type": workflow_type,
            "workflow_data": workflow_data
        }

        return {
            "success": True,
            "call_sid": call.sid,
            "workflow_type": workflow_type,
            "workflow_data": workflow_data
        }

    except Exception as e:
        logger.error(f"❌ Outbound call error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/recording")
async def recording_callback(request: Request):
    """
    Handle Twilio recording callback
    Called when call recording is available
    """
    # Twilio sends form data, not JSON
    try:
        form_data = await request.form()
        call_sid = form_data.get("CallSid")
        recording_url = form_data.get("RecordingUrl")
        recording_sid = form_data.get("RecordingSid")
        
        logger.info(f"🎙️ Recording available for call {call_sid}")
        logger.info(f"   Recording SID: {recording_sid}")
        logger.info(f"   Recording URL: {recording_url}")
        
        # Here you can:
        # - Download the recording
        # - Store metadata in database
        # - Trigger post-call processing
        # - Send notifications
        
        return {"success": True}
    except Exception as e:
        logger.error(f"❌ Recording callback error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/api/calls")
async def list_active_calls():
    """List all active calls"""
    calls = []

    for call_sid, agent in active_calls.items():
        # Skip workflow-only entries
        if isinstance(agent, dict):
            calls.append({
                "call_sid": call_sid,
                "workflow_type": agent.get("workflow_type"),
                "waiting_for_twilio_stream": True
            })
            continue

        calls.append({
            "call_sid": call_sid,
            "stream_sid": agent.stream_sid,
            "questions_asked": agent.question_number,
            "transcripts": agent.total_transcripts,
            "responses": agent.total_responses,
            "is_speaking": agent.is_speaking,
            "awaiting_response": agent.awaiting_response,
        })

    return {
        "active_calls": len(calls),
        "calls": calls
    }

@app.post("/api/calls/{call_sid}/end")
async def end_call(call_sid: str):
    """Manually end a specific call"""
    agent = active_calls.get(call_sid)
    
    if not agent:
        return {"success": False, "error": "Call not found"}
    
    logger.info(f"🛑 Manually ending call {call_sid}")
    
    await agent.end_call()
    
    return {"success": True, "call_sid": call_sid}


if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "=" * 70)
    print("🚀 SARVAM VOICE AGENT - PRODUCTION READY")
    print("=" * 70)
    print()
    print("✅ Sarvam AI Speech-to-Text (WebSocket streaming)")
    print("✅ Sarvam AI Text-to-Speech (WebSocket streaming)")
    print("✅ Azure OpenAI for conversation AI")
    print("✅ Twilio for telephony integration")
    print("✅ Real-time audio processing (μ-law ↔ PCM)")
    print("✅ Interruption handling and turn-taking")
    print()
    print(f"📡 Server starting on {Config.HOST}:{Config.PORT}")
    print(f"🌐 Webhook URL: {Config.WEBHOOK_BASE_URL}")
    print()
    print("=" * 70)
    print()
    
    uvicorn.run(
        app,
        host=Config.HOST,
        port=Config.PORT,
        log_level=Config.LOG_LEVEL.lower()
    )
    
    
    
    #upload routes handler
    
    
from azurestorage import (
    delete_file_from_azure,
    upload_file_to_azure,
    validate_pdf_file,
    generate_unique_filename,
    azure_config,
    ResumeUploadResponse,
    JobDescriptionUploadResponse
)


from typing import List, Optional

from fastapi import FastAPI, HTTPException, File, UploadFile

@app.post("/upload-resumes", response_model=ResumeUploadResponse)
async def upload_resumes(
    resumes: List[UploadFile] = File(..., description="Resume PDF files")
):
    """
    Upload resume PDF files to S3 bucket
    
    Args:
        resumes: List of resume PDF files
    
    Returns:
        JSON response with downloadable URLs for uploaded resume files
    """
    
    # Validate that files are provided
    if not resumes:
        raise HTTPException(
            status_code=400, 
            detail="Resume files are required"
        )
    
    # Validate file types
    # for file in resumes:
    #     if not validate_pdf_file(file):
    #         raise HTTPException(
    #             status_code=400,
    #             detail=f"Only PDF files are allowed. Invalid file: {file.filename}"
    #         )
    
 
    
    
    
    uploaded_resume_urls = []
    
    try:
        # Upload resume files
        for resume_file in resumes:
            # Reset file pointer to beginning
            await resume_file.seek(0)
            url = await upload_file_to_azure(resume_file, "resumes")
            uploaded_resume_urls.append(url)
        
        return ResumeUploadResponse(resumes=uploaded_resume_urls)
        
    except Exception as e:
        # Clean up uploaded files if there's an error
        print(f"Error during resume upload: {str(e)}")
        
        # Extract S3 keys from URLs and delete files
        for url in uploaded_resume_urls:
            try:
                # Extract S3 key from URL
                blob_path = url.split(f"{azure_config.container_name}/")[-1]
                delete_file_from_azure(blob_path)
            except Exception as cleanup_error:
                print(f"Error during cleanup: {str(cleanup_error)}")
        
        raise HTTPException(status_code=500, detail=f"Resume upload failed: {str(e)}")

@app.post("/upload-job-descriptions", response_model=JobDescriptionUploadResponse)

async def upload_job_descriptions(
    job_description: UploadFile = File(..., description="Job description PDF file")
):
    """
    Upload a single job description PDF file to S3 bucket
    
    Args:
        job_description: Single job description PDF file
    
    Returns:
        JSON response with downloadable URL for uploaded job description file
    """
    
    # Validate that file is provided
    if not job_description:
        raise HTTPException(
            status_code=400, 
            detail="Job description file is required"
        )
    
    # Validate file type
    if not validate_pdf_file(job_description):
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are allowed. Invalid file: {job_description.filename}"
        )
    
   
    
    try:
        # Upload job description file
        await job_description.seek(0)
        url = await upload_file_to_azure(job_description, "job_descriptions")
        
        return JobDescriptionUploadResponse(job_descriptions=url)
        
    except Exception as e:
        # Clean up uploaded file if there's an error
        print(f"Error during job description upload: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Job description upload failed: {str(e)}")


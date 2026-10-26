"""The reference engine: Pipecat 1.13 doing chirp's job, for side-by-side benchmarks.

    uv run --project reference python reference/pipecat_bot.py <port> <mocks host:port>

Serves `ws://127.0.0.1:<port>/v1/calls` like chirp: 16 kHz mono PCM16 in and
out as binary frames. One Pipecat pipeline per connection, all in one process,
with Pipecat's defaults for turn-taking (Silero VAD, stop_secs 0.2, then
LocalSmartTurnAnalyzerV3), ElevenLabs realtime STT (manual commit) and
streaming TTS, and an OpenAI-compatible LLM, all pointed at chirp's mocks.
"""

import sys

import uvicorn
from fastapi import FastAPI, WebSocket
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import Frame, InputAudioRawFrame, OutputAudioRawFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.serializers.base_serializer import FrameSerializer
from pipecat.services.elevenlabs.stt import ElevenLabsRealtimeSTTService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.websocket.fastapi import FastAPIWebsocketParams, FastAPIWebsocketTransport

PORT, MOCKS = int(sys.argv[1]), sys.argv[2]


class RawPcm(FrameSerializer):
    """Binary frames are 16 kHz mono PCM16, both ways; nothing else is sent."""

    async def serialize(self, frame: Frame):
        return frame.audio if isinstance(frame, OutputAudioRawFrame) else None

    async def deserialize(self, data):
        if isinstance(data, bytes):
            return InputAudioRawFrame(audio=data, sample_rate=16000, num_channels=1)
        return None


class MockSTT(ElevenLabsRealtimeSTTService):
    """Pipecat builds a wss:// URL; the local mock speaks plain ws://."""

    async def _websocket_connect(self, url, **kwargs):
        return await super()._websocket_connect(url.replace("wss://", "ws://"), **kwargs)


app = FastAPI()


@app.websocket("/v1/calls")
async def call(websocket: WebSocket):
    await websocket.accept()
    transport = FastAPIWebsocketTransport(
        websocket,
        FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_in_sample_rate=16000,
            audio_out_enabled=True,
            audio_out_sample_rate=16000,
            serializer=RawPcm(),
        ),
    )
    stt = MockSTT(api_key="mock", base_url=MOCKS, sample_rate=16000)
    tts = ElevenLabsTTSService(api_key="mock", voice_id="mock", url=f"ws://{MOCKS}", sample_rate=16000)
    llm = OpenAILLMService(api_key="mock", base_url=f"http://{MOCKS}/v1", model="mock")
    context = LLMContext()
    user, assistant = LLMContextAggregatorPair(
        context, user_params=LLMUserAggregatorParams(vad_analyzer=SileroVADAnalyzer())
    )
    pipeline = Pipeline([transport.input(), stt, user, llm, tts, transport.output(), assistant])
    task = PipelineTask(pipeline, params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=16000))

    @transport.event_handler("on_client_disconnected")
    async def on_disconnected(transport, client):
        await task.cancel()

    await PipelineRunner(handle_sigint=False).run(task)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")

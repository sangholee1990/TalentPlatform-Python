# ============================================
# 요구사항
# ============================================
# 챗봇 RAG API

# =================================================
# 도움말
# =================================================
# 프로그램 시작
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2026/trouShoot
# conda activate py311

# 운영 서버
# nohup /HDD/SYSTEMS/LIB/anaconda3/envs/py311/bin/uvicorn TalentPlatform-BDWIDE2026-DaemonApi-chatTrouShoot:app --reload --host=0.0.0.0 --port=9940 &
# tail -f nohup.out

# 테스트 서버
# /HDD/SYSTEMS/LIB/anaconda3/envs/py311/bin/uvicorn TalentPlatform-BDWIDE2026-DaemonApi-chatTrouShoot:app --reload --host=0.0.0.0 --port=9940
# /HDD/SYSTEMS/LIB/anaconda3/envs/py311/bin/python -m uvicorn TalentPlatform-BDWIDE2026-DaemonApi-chatTrouShoot:app --host=0.0.0.0 --port=9940

# 프로그램 종료
# pkill -f TalentPlatform-BDWIDE2026-DaemonApi-chatTrouShoot
# ps -ef | grep "TalentPlatform-BDWIDE2026-DaemonApi-chatTrouShoot" | awk '{print $2}' | xargs kill -9

# 포트 종료
# yum install lsof -y
# lsof -i :9940
# lsof -i :9940 | awk '{print $2}' | xargs kill -9

# llama-server 시작
# cd D:\ollama\llama-b10502-bin-win-cpu-x64
# llama-server.exe -m "D:/ollama/gemma-4-E2B-it-Q4_K_M.gguf" --mmproj "D:/ollama/mmproj-F16.gguf" --host 0.0.0.0 --port 9941 -ngl 999 -c 4096 --parallel 2 -fa on -rea off

# export LD_LIBRARY_PATH=/HDD/SYSTEMS/LIB/anaconda3/envs/py311/lib:$LD_LIBRARY_PATH
# cd /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2026/trouShoot/llama-b10502
# /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2026/trouShoot/llama-b10502/llama-server -m "/HDD/DATA/INPUT/BDWIDE2026/chatTrouShoot/ollama/gemma-4-E2B-it-Q4_K_M.gguf" --mmproj "/HDD/DATA/INPUT/BDWIDE2026/chatTrouShoot/ollama/mmproj-F16.gguf" --host 0.0.0.0 --port 9941 -ngl 999 -c 4096 --parallel 3 -fa on -rea off
# /HDD/SYSTEMS/PROG/PYTHON/IDE/src/proj/bdwide/2026/trouShoot/llama-b10502/llama-server -m "/HDD/DATA/INPUT/BDWIDE2026/chatTrouShoot/ollama/gemma-4-E2B-it-Q4_K_M.gguf" --host 0.0.0.0 --port 9941 -ngl 999 -c 4096 --parallel 2 -fa on -rea off

# ============================================
# 라이브러리
# ============================================
import os
import platform
import logging
import logging.handlers
import sys
import traceback
import warnings
import pytz
import time
import threading
import json
import asyncio
import queue
from typing import Any, List
from fastapi import FastAPI, Depends, HTTPException, Form, Query, Request, WebSocket, WebSocketDisconnect
from fastapi import FastAPI, Depends, HTTPException, Form, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[ChatMessage]
    stream: bool = False
    temperature: float = 0.5
    max_tokens: int = -1

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Windows 인증서 스토어 오류(ASN1: NOT_ENOUGH_DATA) 및 깨진 환경변수 우회 패치 ---
import ssl
import certifi
import os

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

orig_create_default_context = ssl.create_default_context
def create_default_context_patched(*args, **kwargs):
    kwargs['cafile'] = certifi.where()
    return orig_create_default_context(*args, **kwargs)
ssl.create_default_context = create_default_context_patched
# ---------------------------------------------------------------------

from openai import AsyncOpenAI

warnings.filterwarnings('ignore')

# ============================================
# 유틸리티 함수
# ============================================
# 로그 설정
def initLog(env=None, contextPath=None, prjName=None):
    if env is None: env = 'local'
    if contextPath is None: contextPath = os.getcwd()
    if prjName is None: prjName = 'test'

    saveLogFile = "{}/{}_{}_{}_{}_{}.log".format(
        contextPath if env in 'local' else os.path.join(contextPath, 'resources', 'log', prjName)
        , platform.system()
        , platform.machine()
        , platform.architecture()[0]
        , platform.node()
        , prjName
    )

    os.makedirs(os.path.dirname(saveLogFile), exist_ok=True)

    log = logging.getLogger(prjName)

    if len(log.handlers) > 0:
        return log

    format = logging.Formatter('%(asctime)s [%(name)s | %(lineno)d | %(filename)s] [%(levelname)-5.5s] %(message)s')

    streamHandler = logging.StreamHandler()
    fileHandler = logging.handlers.TimedRotatingFileHandler(filename=saveLogFile, when='midnight', interval=1, backupCount=30, encoding='utf-8')

    streamHandler.setFormatter(format)
    fileHandler.setFormatter(format)

    log.addHandler(streamHandler)
    log.addHandler(fileHandler)

    log.setLevel(level=logging.INFO)

    return log

# 인증키 검사
def chkKey(key: str = Depends(APIKeyHeader(name="key"))):
    if key != '20260221-bdwide':
        raise HTTPException(status_code=400, detail="API 인증 실패")

def resResponse(status: str, code: int, message: str, cnt: Any = None, data: Any = None) -> dict:
    return {
        "status": status
        , "code": code
        , "message": message
        , "cnt": cnt
        , "data": data
    }

# ============================================
# 주요 설정
# ============================================
env = 'dev'
serviceName = 'BDWIDE2026'
prjName = 'chatTrouShoot'

if platform.system() == 'Windows':
    ctxPath = os.getcwd() if env in 'local' else 'C:/SYSTEMS/PROG/PYTHON/TalentPlatform-Python'
else:
    ctxPath = os.getcwd() if env in 'local' else '/HDD/SYSTEMS/PROG/PYTHON/IDE'

log = initLog(env, ctxPath, prjName)

# 옵션 설정
sysOpt = {
    'oriList': ['*'],
    # 'embModel': 'D:/ollama/multilingual-e5-small',
    # 'vecDb': 'D:/ollama/trouShoot_chromadb',
    'embModel': '/HDD/DATA/INPUT/BDWIDE2026/chatTrouShoot/ollama/multilingual-e5-small',
    'vecDb': '/HDD/DATA/INPUT/BDWIDE2026/chatTrouShoot/ollama/trouShoot_chromadb',
}

app = FastAPI(
    title="AI 트러블슈팅 챗봇 API",
    description="문서 기반 AI 챗봇 질의응답 API",
    version="1.0",
    openapi_url='/api',
    docs_url='/docs',
    redoc_url='/redoc',
)

app.add_middleware(
    CORSMiddleware
    , allow_origins=sysOpt['oriList']
    , allow_credentials=True
    , allow_methods=["*"]
    , allow_headers=["*"]
)

# ============================================
# 비즈니스 로직
# ============================================
# 모델 초기화
try:
    log.info("로딩 중: 임베딩 모델 및 ChromaDB 불러오기...")
    embeddings = HuggingFaceEmbeddings(
        model_name=sysOpt['embModel'],
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    vectorstore = Chroma(
        persist_directory=sysOpt['vecDb'],
        embedding_function=embeddings
    )

    log.info("로딩 중: LLM 클라이언트 설정...")
    llm_client = AsyncOpenAI(
        api_key="llama-cpp",
        base_url="http://localhost:9941/v1"
    )
    log.info("LLM 클라이언트 연결 완료")
except Exception as e:
    import traceback
    log.error(f"Exception during model load : {e}")
    log.error(traceback.format_exc())
    sys.exit(1)

# ============================================
# API URL 주소
# ============================================
@app.get(f"/", response_class=HTMLResponse, include_in_schema=False)
async def get_chat_page():
    html_content = """
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>AI 트러블슈팅 챗봇</title>
        <!-- Pretendard Font -->
        <link rel="stylesheet" as="style" crossorigin href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard-gov.min.css" />
        <script src="https://cdn.tailwindcss.com?plugins=typography"></script>
        <script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
        <script>
            tailwind.config = {
                theme: {
                    extend: {
                        fontFamily: { sans: ['"Pretendard Gov"', 'Pretendard', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'] },
                        colors: {
                            geminiBg: '#ffffff',
                            geminiUserBg: '#f0f4f9',
                            geminiText: '#1f1f1f'
                        },
                        animation: {
                            'fade-in-down': 'fadeInDown 0.8s ease forwards',
                            'fade-in-up': 'fadeInUp 0.8s ease forwards',
                            'pop-in': 'popIn 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) forwards',
                            'typing': 'typing 1.4s infinite ease-in-out both'
                        },
                        keyframes: {
                            fadeInDown: { '0%': { opacity: '0', transform: 'translateY(-20px)' }, '100%': { opacity: '1', transform: 'translateY(0)' } },
                            fadeInUp: { '0%': { opacity: '0', transform: 'translateY(20px)' }, '100%': { opacity: '1', transform: 'translateY(0)' } },
                            popIn: { '0%': { opacity: '0', transform: 'scale(0.95) translateY(10px)' }, '100%': { opacity: '1', transform: 'scale(1) translateY(0)' } },
                            typing: { '0%, 80%, 100%': { transform: 'scale(0)' }, '40%': { transform: 'scale(1)' } }
                        }
                    }
                }
            }
        </script>
        <style>
            body { background-color: #ffffff; }
            #chat-box::-webkit-scrollbar { width: 8px; }
            #chat-box::-webkit-scrollbar-thumb { background: rgba(0,0,0,0.1); border-radius: 10px; }
            #chat-box::-webkit-scrollbar-thumb:hover { background: rgba(0,0,0,0.2); }
            .delay-1 { animation-delay: -0.32s; }
            .delay-2 { animation-delay: -0.16s; }
            
            /* 마크다운 스타일 Gemini 톤 커스텀 */
            .prose p { line-height: 1.7; color: #1f1f1f; }
            .prose strong { color: #1f1f1f; font-weight: 600; }
            .prose a { color: #0b57d0; text-decoration: none; }
            .prose a:hover { text-decoration: underline; }
            .prose pre { background-color: #f0f4f9 !important; color: #1f1f1f !important; border-radius: 16px !important; }
            .prose code { color: #1f1f1f; background-color: #f0f4f9; padding: 2px 6px; border-radius: 6px; }
        </style>
    </head>
    <body class="bg-white text-[#1f1f1f] h-screen flex flex-col overflow-hidden font-sans relative">
        
        <!-- Header -->
        <div class="absolute top-0 left-0 w-full p-4 md:p-6 flex items-center justify-between z-10 bg-gradient-to-b from-white via-white to-transparent">
            <h2 class="text-xl md:text-2xl font-normal tracking-tight text-[#1f1f1f] flex items-center gap-2">
                <span class="bg-gradient-to-r from-blue-500 to-purple-500 bg-clip-text text-transparent font-medium">✨ AI 트러블슈팅 챗봇</span>
            </h2>
        </div>
        
        <!-- Chat Area -->
        <div id="chat-box" class="flex-1 w-full overflow-y-auto flex flex-col items-center pt-24 pb-32 scroll-smooth px-4">
            <div id="chat-inner" class="w-full max-w-[800px] flex flex-col gap-8 md:gap-10"></div>
        </div>
        
        <!-- Input Area -->
        <div class="absolute bottom-0 left-0 w-full bg-gradient-to-t from-white via-white to-transparent pt-10 pb-6 px-4 pointer-events-none">
            <div class="max-w-[800px] mx-auto pointer-events-auto flex flex-col gap-2">
                <div class="flex items-center justify-end px-2">
                    <div class="flex items-center gap-2 bg-[#f0f4f9] px-3 py-1.5 rounded-full border border-gray-100 shadow-sm">
                        <label for="temp-slider" class="text-[0.75rem] text-gray-600 font-medium whitespace-nowrap">Temperature 창의성<span id="temp-val" class="font-bold w-5 inline-block text-right">0.5</span></label>
                        <input type="range" id="temp-slider" min="0.0" max="1.0" step="0.1" value="0.5" oninput="document.getElementById('temp-val').innerText=Number(this.value).toFixed(1)" class="w-24 h-1.5 bg-gray-300 rounded-lg appearance-none cursor-pointer outline-none accent-blue-500">
                    </div>
                </div>
                <form id="chat-form" onsubmit="sendMessage(event)" class="relative flex items-end gap-2 bg-[#f0f4f9] rounded-[32px] px-2 py-2 md:p-2 min-h-[60px] focus-within:ring-1 focus-within:ring-gray-300 transition-all">
                    <input type="text" id="user-input" placeholder="여기에 프롬프트를 입력하세요" required autocomplete="off" 
                           class="flex-1 bg-transparent border-none text-[#1f1f1f] text-base md:text-[1.05rem] px-4 py-3 focus:outline-none focus:ring-0 placeholder:text-gray-500 min-h-[48px]">
                    <button type="submit" 
                            class="w-12 h-12 rounded-full bg-white hover:bg-gray-100 flex items-center justify-center transition-colors text-black shrink-0 shadow-sm border border-gray-100">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
                    </button>
                </form>
                <div class="text-center mt-3 text-[0.75rem] text-gray-500 font-medium">
                    AI 챗봇은 실수를 할 수 있습니다. 중요한 정보를 확인하세요.
                </div>
            </div>
        </div>

        <script>
            let ws = null;

            async function sendMessage(event) {
                event.preventDefault();
                const input = document.getElementById('user-input');
                const chatBox = document.getElementById('chat-box');
                const query = input.value;
                if (!query.trim()) return;
                input.value = '';
                
                let fullAiResponse = '';

                const chatInner = document.getElementById('chat-inner');
                
                // 유저 메시지 렌더링
                const userWrapper = document.createElement('div');
                userWrapper.className = 'flex flex-col w-full items-end animate-pop-in opacity-0';
                userWrapper.innerHTML = `
                    <div class="max-w-[90%] md:max-w-[75%] px-6 py-4 rounded-[24px] text-[0.95rem] md:text-base leading-relaxed bg-[#f0f4f9] text-[#1f1f1f] break-words">
                        ${query}
                    </div>
                `;
                chatInner.appendChild(userWrapper);

                // AI 응답 렌더링
                const aiWrapper = document.createElement('div');
                aiWrapper.className = 'flex w-full items-start gap-4 md:gap-6 animate-fade-in-up opacity-0 mt-2';
                aiWrapper.innerHTML = `
                    <div class="w-8 h-8 rounded-full shrink-0 flex items-center justify-center mt-1">
                        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" class="text-blue-500"><path d="M12 2L14.64 9.36L22 12L14.64 14.64L12 22L9.36 14.64L2 12L9.36 9.36L12 2Z" fill="currentColor"/></svg>
                    </div>
                    <div class="flex-1 min-w-0">
                        <div class="text-[0.95rem] md:text-base leading-relaxed break-words bg-transparent ai-text text-[#1f1f1f] prose prose-slate max-w-none prose-p:my-2 prose-ul:my-2">
                            <div class="inline-flex items-center gap-1 h-6"><span class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-typing delay-1"></span><span class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-typing delay-2"></span><span class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-typing"></span></div>
                        </div>
                    </div>
                `;
                chatInner.appendChild(aiWrapper);
                const currentAiElement = aiWrapper.querySelector('.ai-text');
                chatBox.scrollTop = chatBox.scrollHeight;

                if (ws) ws.close();
                const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/chatTrouShoot`);

                let isDone = false;
                let hasReceivedContent = false;

                ws.onopen = () => {
                    const temp = parseFloat(document.getElementById('temp-slider').value) || 0.5;
                    ws.send(JSON.stringify({query: query, temperature: temp}));
                    fullAiResponse = '';
                };

                ws.onmessage = (event) => {
                    if (event.data === '[DONE]') {
                        isDone = true;
                        if (!hasReceivedContent) {
                            currentAiElement.innerHTML = `<span class="text-gray-500 text-sm md:text-base">검색된 결과가 없습니다.</span>`;
                        }
                        return;
                    }
                    try {
                        const data = JSON.parse(event.data);
                        if (data.clear) {
                            fullAiResponse = '';
                            currentAiElement.innerHTML = '<div class="inline-flex items-center gap-1 h-6"><span class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-typing delay-1"></span><span class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-typing delay-2"></span><span class="w-1.5 h-1.5 bg-gray-400 rounded-full animate-typing"></span></div>';
                            hasReceivedContent = false;
                            return;
                        }
                        if (data.error) {
                            if (!hasReceivedContent) currentAiElement.innerHTML = '';
                            hasReceivedContent = true;
                            currentAiElement.innerHTML += `<br><span class="text-red-500">${data.error}</span>`;
                            chatBox.scrollTop = chatBox.scrollHeight;
                            return;
                        }
                        if (data.content) {
                            if (!hasReceivedContent) currentAiElement.innerHTML = '';
                            hasReceivedContent = true;
                            fullAiResponse += data.content;
                            currentAiElement.innerHTML = marked.parse(fullAiResponse, { breaks: true });
                            chatBox.scrollTop = chatBox.scrollHeight;
                            return;
                        }
                        if (data.choices && data.choices[0].delta && data.choices[0].delta.content) {
                            if (!hasReceivedContent) currentAiElement.innerHTML = '';
                            hasReceivedContent = true;
                            fullAiResponse += data.choices[0].delta.content;
                            try {
                                currentAiElement.innerHTML = marked.parse(fullAiResponse, { breaks: true });
                            } catch (err) {
                                currentAiElement.innerHTML = fullAiResponse.replace(/\\n/g, '<br>');
                            }
                            chatBox.scrollTop = chatBox.scrollHeight;
                        }
                    } catch (e) {
                        console.log("WebSocket Raw Message:", event.data);
                    }
                };

                ws.onerror = (error) => {
                    console.error("WebSocket 오류:", error);
                    currentAiElement.innerHTML = `<span class="text-red-500">서버와의 연결이 끊어졌습니다.</span>`;
                };
                
                ws.onclose = () => {
                    if (!isDone && !hasReceivedContent && !currentAiElement.innerHTML.includes('text-red-500')) {
                        currentAiElement.innerHTML = `<span class="text-gray-500 text-sm md:text-base">서버에서 응답을 반환하지 않았습니다.</span>`;
                    }
                };
            }
        </script>
    </html>
    """
    return html_content

# ============================================
# 일반 테스트용 파라미터 API (Swagger UI 노출)
# ============================================
@app.post(f"/api/chatTrouShoot")
async def chatTrouShoot(query: str = Query(..., description="사용자 질문")):
    """
    기능\n
        문서 기반 챗봇 질의응답 (RAG) 스트리밍 API\n
    파라미터\n
        query: 사용자 질문 (GET 방식 또는 POST 방식 모두 지원)\n
    """
    try:
        start_time = time.time()

        if not query or not query.strip():
            return JSONResponse(status_code=400, content={"error": "질문을 찾을 수 없습니다."})

        if vectorstore is None:
            return JSONResponse(status_code=500, content={"error": "서버 모델이 로드되지 않았습니다."})

        # 1단계: Context Retrieval
        docs = vectorstore.similarity_search(query, k=2)
        retrieved_context = "\\n\\n".join([doc.page_content for doc in docs])

        # 2단계: RAG 프롬프트 구성
        prompt_text = f'''
        다음 제공된 [참고 문서]를 바탕으로 원인, 우선 확인사항, 조치사항 등에 답변해 줘

        [지시사항]
        1. 각 항목(원인, 확인사항 등)마다 최소 1~3줄 이상 길고 구체적으로 설명할 것.
        2. 단순히 문서를 요약하지 말고, 실무자가 바로 이해하고 적용할 수 있도록 가이드라인 형태로 작성할 것.

        [참고 문서]
        {retrieved_context}

        [질문]
        {query}
        '''

        llama_messages = [
            {"role": "system", "content": "당신은 제공된 문서를 바탕으로 분석하는 친절하고 전문적인 AI 어시스턴트입니다."},
            {"role": "user", "content": prompt_text.strip()}
        ]

        log.info(f"파라미터 방식 스트리밍 질의: {query}")

        # 동기(Non-streaming) 응답
        log.info(f"파라미터 방식 단일 질의 처리 중...")
        response = await llm_client.chat.completions.create(
            model="default",
            messages=llama_messages,
            temperature=0.5,
            stream=False
        )
        
        # AsyncOpenAI 반환 객체를 dict로 변환
        resp_dict = response.model_dump()
        log.info(f"단일 응답 완료. 소요 시간: {time.time() - start_time:.2f}초")
        return JSONResponse(content=resp_dict)

    except Exception as e:
        log.error(f'Exception : {e}')
        log.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error": f"답변 생성 실패: {str(e)}"})


# ============================================
# WebSocket 방식 실시간 스트리밍 API (궁극적 타자 효과)
# ============================================
@app.websocket("/ws/chatTrouShoot")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        data = await websocket.receive_text()
        try:
            req_data = json.loads(data)
            query = req_data.get("query", "")
            temperature = float(req_data.get("temperature", 0.5))
        except:
            query = data
            temperature = 0.5

        if not query.strip():
            await websocket.send_text(json.dumps({"content": "질문을 입력해주세요."}, ensure_ascii=False))
            await websocket.close()
            return

        if vectorstore is None:
            await websocket.send_text(json.dumps({"content": "서버 모델이 로드되지 않았습니다."}, ensure_ascii=False))
            await websocket.close()
            return

        docs = vectorstore.similarity_search(query, k=2)
        retrieved_context = "\\n\\n".join([doc.page_content for doc in docs])

        prompt_text = f'''
        다음 제공된 [참고 문서]를 바탕으로 원인, 우선 확인사항, 조치사항 등에 답변해 줘

        [지시사항]
        1. 각 항목(원인, 확인사항 등)마다 최소 1~3줄 이상 길고 구체적으로 설명할 것.
        2. 단순히 문서를 요약하지 말고, 실무자가 바로 이해하고 적용할 수 있도록 가이드라인 형태로 작성할 것.

        [참고 문서]
        {retrieved_context}

        [질문]
        {query}
        '''

        llama_messages = [
            {"role": "system", "content": "당신은 제공된 문서를 바탕으로 분석하는 친절하고 전문적인 AI 어시스턴트입니다."},
            {"role": "user", "content": prompt_text.strip()}
        ]

        log.info(f"웹소켓 질의: {query}")

        try:
            response_stream = await llm_client.chat.completions.create(
                model="default",
                messages=llama_messages,
                temperature=temperature,
                stream=True
            )
            
            async for chunk in response_stream:
                if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                    await websocket.send_text(json.dumps(chunk.model_dump(), ensure_ascii=False))
                    
            await websocket.send_text("[DONE]")
            await websocket.close()
            
        except WebSocketDisconnect:
            log.info("클라이언트가 스트리밍 도중 연결을 끊었습니다.")
        except Exception as e:
            log.error(f"LLM Error: {e}")
            try:
                await websocket.send_text(json.dumps({"error": f"오류 발생: {e}"}, ensure_ascii=False))
            except RuntimeError:
                pass
    except WebSocketDisconnect:
        log.info("웹소켓 클라이언트 연결 끊김")
    except Exception as e:
        log.error(f'WebSocket Exception : {e}')
        log.error(traceback.format_exc())
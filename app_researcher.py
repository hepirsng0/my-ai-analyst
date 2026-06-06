import os
import streamlit as st
from typing import Dict, TypedDict, List
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilyAnswer
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# 🔑 API 키 설정
GROQ_API_KEY = "GROQ_API_KEY"
TAVILY_API_KEY = "TAVILY_API_KEY"  # 👈 본인의 테빌리 키를 꼭 넣어주세요!

os.environ["TAVILY_API_KEY"] = TAVILY_API_KEY

# 1. LLM 및 검색 도구 선언
llm = ChatOpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
    temperature=0.2
)
web_search_tool = TavilyAnswer(max_results=3)

# 2. LangGraph 상태 정의
class ResearcherState(TypedDict):
    company_name: str     
    search_keywords: str  
    search_results: str   
    final_report: str     
    loop_count: int       
    messages: List[BaseMessage]  # 💬 대화 기록이 유실되지 않고 Append 되도록 유지

# 3. LangGraph 노드 및 에지 함수들
def plan_research_node(state: ResearcherState) -> Dict:
    company = state.get("company_name", "")
    current_count = state.get("loop_count", 0)
    chat_history = state.get("messages", [])
    
    # 🧠 이미 대화가 시작된 상태(추가 질문)라면 구글 검색어 생성을 SKIP 합니다.
    if not company and len(chat_history) > 0:
        return {"search_keywords": "SKIP"}

    extra_instruction = ""
    if current_count > 0:
        extra_instruction = f"⚠️ 이전 검색이 부실했습니다. 이번에는 더 구체적인 키워드를 섞어주세요. 이전 키워드: {state.get('search_keywords', '')}"

    prompt = f"'{company}' 회사에 대한 최신 뉴스와 주가 동향을 구글에 검색하려고 합니다. 가장 정확한 최신 정보를 긁어올 수 있는 검색어 딱 1개를 문장이나 단어로 적어주세요. 다른 말은 절대 하지 말고 오직 검색어만 출력하세요.\n{extra_instruction}"
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "search_keywords": str(response.content).strip().replace('"', ''),
        "loop_count": current_count + 1
    }

def web_search_node(state: ResearcherState) -> Dict:
    keywords = state["search_keywords"]
    if keywords == "SKIP":
        return {"search_results": "SKIP"}
        
    try:
        raw_search_result = web_search_tool.invoke({"query": keywords})
        search_data = str(raw_search_result).strip()
    except Exception:
        search_data = f"검색 실패로 인해 데이터가 부족합니다. {keywords}에 대한 정보 없음."
    return {"search_results": search_data}

def evaluate_research_edge(state: ResearcherState) -> str:
    keywords = state["search_keywords"]
    if keywords == "SKIP":
        return "chat_mode" # 추가 질문(대화 모드)일 때는 바로 보고서/답변 노드로 워프!

    results = state["search_results"]
    current_count = state["loop_count"]
    
    if current_count >= 2:
        return "sufficient"
    if "검색 실패" in results or len(results) < 40:
        return "insufficient"
    return "sufficient"

def write_report_node(state: ResearcherState) -> Dict:
    company = state.get("company_name", "")
    results = state["search_results"]
    
    # 🧠 기존 세션에 누적되어 온 대화 기록을 가져옵니다. (없으면 빈 리스트)
    chat_history = state.get("messages", []) or []

    system_instruction = (
        "You are a professional financial investment analyst. "
        "반드시 한국어로 답변하세요. 대화 기록(chat history)과 제공된 실시간 데이터에 기반하여 답변해야 합니다.\n"
        "⚠️ 절대 '[기업명]', '[주가]', '[분야/업종]' 같은 괄호 껍데기나 빈 양식 템플릿 형태로 답변을 내보내지 마세요. "
        "사용자가 이전 대화에 이어 추가 질문을 던진 경우, 과거에 분석했던 기업 정보와 맥락을 온전히 기억하여 구체적이고 실질적인 답변과 투자 조언을 제공하세요."
    )

    # 대화 주머니 빌드업
    messages = [SystemMessage(content=system_instruction)]
    
    # 🧠 과거 대화 맥락이 LLM에게 전달되도록 히스토리를 먼저 통째로 주입!
    if chat_history:
        messages.extend(chat_history)

    # 처음 분석하는 거라면 실시간 구글 검색 결과 텍스트를 묶어서 전달
    if results != "SKIP" and results:
        messages.append(HumanMessage(content=f"실시간 인터넷 검색 정보:\n{results}\n\n위 데이터를 바탕으로 분석 보고서를 작성해줘."))

    # LLM 실행
    response = llm.invoke(messages)
    new_ai_message = AIMessage(content=str(response.content).strip())
    
    # 🧠 중요한 대화 내역 누적 처리: 기존 역사 뒤에 이번 AI 답변을 붙여서 업데이트
    updated_messages = chat_history + [new_ai_message]
    
    return {
        "final_report": str(response.content).strip(),
        "messages": updated_messages
    }

# 4. LangGraph 그래프 조립 (Memory 가동)
workflow = StateGraph(ResearcherState)
workflow.add_node("plan_research", plan_research_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("write_report", write_report_node)

workflow.set_entry_point("plan_research")
workflow.add_edge("plan_research", "web_search")
workflow.add_conditional_edges(
    "web_search",
    evaluate_research_edge,
    {
        "sufficient": "write_report", 
        "insufficient": "plan_research",
        "chat_mode": "write_report"
    }
)
workflow.add_edge("write_report", END)

# 체크포인터 메모리 매핑 후 컴파일
memory = MemorySaver()
research_agent = workflow.compile(checkpointer=memory)


# 5. 🎨 UI 디자인 (Streamlit 챗봇 형식)
st.set_page_config(page_title="AI 금융 애널리스트 챗봇", page_icon="💬", layout="wide")

st.title("💬 Memory가 탑재된 AI 금융 애널리스트")
st.caption("Hayden님의 세 번째 심화 프로젝트: 대화 흐름을 완벽히 기억하는 리서치 에이전트")

# 화면 유지용 세션 상태 초기화
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []

# 🧠 LangGraph 메모리가 대화를 기억할 수 있게 지정하는 방 번호(Thread ID)
config = {"configurable": {"thread_id": "hayden_perfect_room"}}

# 하단 채팅 입력창
user_input = st.chat_input("에이전트에게 기업 분석을 요청하거나 추가 질문을 던져보세요!")

# 이전 대화 기록 화면에 다시 그려주기
for msg in st.session_state["chat_messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 유저가 채팅을 치면 작동 시작
if user_input:
    # 1. 유저 말풍선 즉시 생성
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["chat_messages"].append({"role": "user", "content": user_input})
    
    # 2. 에이전트 연산 상태 연출
    with st.chat_message("assistant"):
        with st.status("🤖 에이전트가 데이터 조율 중...", expanded=False) as status:
            
            # 첫 질문인지 꼬리 질문인지 감지
            is_first = not any(m["role"] == "assistant" for m in st.session_state["chat_messages"][:-1])
            
            if is_first:
                status.update(label=f"🔍 '{user_input}' 정보 실시간 구글링 및 분석 중...")
                inputs = {
                    "company_name": user_input,
                    "search_keywords": "",
                    "search_results": "",
                    "final_report": "",
                    "loop_count": 0,
                    "messages": [HumanMessage(content=user_input)]
                }
            else:
                status.update(label="🧠 이전 대화 흐름 분석 및 투자 제언 작성 중...")
                inputs = {
                    "company_name": "", # 추가 질문일 땐 회사명을 초기화하여 SKIP 유도
                    "messages": [HumanMessage(content=user_input)]
                }
            
            # 방 번호(config)를 동반하여 랭그래프 호출 (메모리 로딩)
            final_output = research_agent.invoke(inputs, config=config)
            status.update(label="✅ 분석 완료!", state="complete")
        
        # 3. 훌륭해진 보고서 양식 출력 및 세션 저장
        with st.container(border=True):
            st.markdown(final_output["final_report"])
            
        st.session_state["chat_messages"].append({"role": "assistant", "content": final_output["final_report"]})
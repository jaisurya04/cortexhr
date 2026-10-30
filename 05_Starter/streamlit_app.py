"""Visual browser-based chat UI for the CortexHR Phase 2 demo."""
from pathlib import Path
import sys
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from cortexhr.phase2.chatbot import train_intent_model, HRChatbot
CSV = ROOT/'03_Data'/'csv'

st.set_page_config(page_title='CortexHR HR Assistant', page_icon='💬', layout='centered')
st.title('💬 CortexHR HR Assistant')
st.caption('Phase 2 · AI/ML Track · Demo using fictional training data. Do not enter real employee data.')
with st.sidebar:
    st.subheader('Demo settings')
    users = __import__('pandas').read_csv(ROOT/'05_Starter'/'mock_services'/'mock_data'/'demo_users.csv').fillna('')
    identities = {f"{r.employee_id} · {r.role}": (r.employee_id, r.role) for _, r in users.iterrows()}
    selected_identity = st.selectbox('Authenticated demo identity (X-Demo-User)', list(identities))
    demo_user, role = identities[selected_identity]
    st.caption(f'Role resolved from mock directory: **{role}**')
    session_id = st.text_input('Session ID', value='demo-session-001')
    st.info('Access is policy-gated. Requests for another employee’s salary are always refused.')
    if st.button('Clear conversation', use_container_width=True):
        st.session_state.messages=[]
        if 'bot' in st.session_state: st.session_state.bot.reset(session_id)
        st.rerun()
@st.cache_resource
def load_bot():
    model = train_intent_model(CSV/'chatbot_nlu_training.csv')
    return HRChatbot(model, CSV/'chatbot_responses.csv', CSV/'hr_faq.csv', users_csv=ROOT/'05_Starter'/'mock_services'/'mock_data'/'demo_users.csv')
if 'bot' not in st.session_state: st.session_state.bot=load_bot()
if 'messages' not in st.session_state: st.session_state.messages=[]
for item in st.session_state.messages:
    with st.chat_message(item['role']): st.markdown(item['content'])
if prompt := st.chat_input('Ask about leave, attendance, payslips or HR policies…'):
    st.session_state.messages.append({'role':'user','content':prompt})
    with st.chat_message('user'): st.markdown(prompt)
    result=st.session_state.bot.reply(prompt, session_id=session_id, user_role=role, demo_user=demo_user)
    answer=result['reply']
    st.session_state.messages.append({'role':'assistant','content':answer})
    with st.chat_message('assistant'):
        st.markdown(answer)
        with st.expander('Debug details (demo)'):
            st.json({'intent':result['intent'],'confidence':round(result['confidence'],3),'slots':result['slots'],'service_called':result['service_called']})

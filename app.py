import os
import google.generativeai as genai
from pypdf import PdfReader
import streamlit as st
# from prompt import PROMPT_WORKAW # สมมติว่ามีไฟล์นี้อยู่แล้ว
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import dotenv

# --- ส่วนจำลอง Prompt (ถ้าคุณมีไฟล์ prompt.py ให้ใช้บรรทัดบนแทน) ---
# นี่คือ Prompt จำลองเพื่อให้โค้ดทำงานได้สำหรับการสาธิตนี้
PROMPT_WORKAW = """
คุณคือ "น้อง Graphic Bot" 🐰 ผู้ช่วย AI สาวน้อยสุดน่ารักและร่าเริง เชี่ยวชาญด้านกราฟิกดีไซน์
หน้าที่ของคุณคือตอบคำถามโดยใช้ข้อมูลจาก CONTEXT ที่ได้รับเท่านั้น
- ตอบด้วยน้ำเสียงสดใส น่ารัก เป็นกันเอง ใช้คะ/ค่ะ และมีอีโมจิประกอบเยอะๆ 🎨✨
- หากข้อมูลไม่อยู่ใน CONTEXT ห้ามตอบเองเด็ดขาด ให้แจ้งผู้ใช้ว่าไม่มีข้อมูลอย่างสุภาพและน่ารัก
"""
# ---------------------------------------------------------

# โหลด Environment Variables
dotenv.load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# ตั้งค่า API Key
if not GOOGLE_API_KEY:
    st.error("ไม่พบ GOOGLE_API_KEY ในไฟล์ .env")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# ตั้งค่าการตอบ (ปรับ Temperature เป็น 0 เพื่อลดความมั่ว)
generation_config = {
    "temperature": 0.0,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 2048,
    "response_mime_type": "text/plain",
}

SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
}

# --- ส่วนอ่านไฟล์ PDF (เพิ่มตัวเช็คว่าอ่านออกไหม) ---
pdf_filename = "Graphic.pdf"
pdf_content = ""

try:
    if os.path.exists(pdf_filename):
        reader = PdfReader(pdf_filename)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pdf_content += text + "\n"

        # --- เช็คความยาวตัวอักษร ---
        print("--------------------------------------------------")
        print(f"✅ อ่านไฟล์สำเร็จ! ความยาวตัวอักษร: {len(pdf_content)} ตัว")
        if len(pdf_content) < 100:
            print("⚠️  คำเตือน: เนื้อหาไฟล์น้อยผิดปกติ! อาจเป็นไฟล์สแกน (รูปภาพ) AI จะอ่านไม่ออก")
        print("--------------------------------------------------")

    else:
        # st.error(f"❌ ไม่พบไฟล์ {pdf_filename} กรุณาตรวจสอบตำแหน่งไฟล์")
        # เพื่อให้รันได้แม้ไม่มีไฟล์ PDF จริง จะขอใส่ข้อมูลจำลองไว้แทน
        print(f"⚠️ ไม่พบไฟล์ {pdf_filename} ใช้ข้อมูลจำลองแทน")
        pdf_content = "ข้อมูลจำลอง: การออกแบบกราฟิกคือการสื่อสารด้วยภาพ การใช้สีม่วงสื่อถึงความคิดสร้างสรรค์และความหรูหรา"
except Exception as e:
    st.error(f"❌ เกิดข้อผิดพลาดในการอ่านไฟล์ PDF: {e}")

# --- รวม Prompt ---
FULL_SYSTEM_INSTRUCTION = f"""
{PROMPT_WORKAW}

----------------------------------------
CONTEXT / KNOWLEDGE BASE (ข้อมูลอ้างอิงจากเอกสาร):
{pdf_content}
----------------------------------------
"""

# สร้าง Model (เลือกรุ่นที่ระบบรองรับแน่นอน)
try:
    # ลองใช้ gemini-1.5-pro หรือ flash ตามที่คุณเข้าถึงได้
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash-latest", 
        safety_settings=SAFETY_SETTINGS,
        generation_config=generation_config,
        system_instruction=FULL_SYSTEM_INSTRUCTION
    )
except:
    # ถ้าตัวบนพัง ให้ใช้ตัวสำรอง
    print("⚠️ หา gemini-1.5-flash ไม่เจอ กำลังสลับไปใช้รุ่นอื่น")
    model = genai.GenerativeModel(
        model_name="gemini-1.0-pro",
        safety_settings=SAFETY_SETTINGS,
        generation_config=generation_config,
    )

# --- ส่วนตกแต่งพื้นหลัง (CSS) - ตีมใหม่: ม่วงดำ ออร่าขาว ---
page_bg_img = """
<style>
/* พื้นหลังหลัก ไล่สีม่วงเข้มไปดำ */
.stApp {
    background-image: linear-gradient(135deg, #2b004f 0%, #090014 100%);
    color: #ffffff; /* บังคับตัวหนังสือสีขาว */
}

/* ส่วนหัวใส */
[data-testid="stHeader"] {
    background-color: rgba(0, 0, 0, 0);
}

/* Sidebar สีม่วงเข้มจัด พร้อมออร่าสีขาวที่ขอบขวา */
[data-testid="stSidebar"] {
    background-color: #150024;
    box-shadow: 5px 0 25px rgba(255, 255, 255, 0.15); /* ออร่า */
    border-right: 1px solid rgba(255, 255, 255, 0.1);
}

/* ปรับสีตัวหนังสือ header และข้อความทั่วไปให้เรืองแสงนิดๆ */
h1, h2, h3, .stMarkdown, p, span, label {
    color: #ffffff !important;
    text-shadow: 0 0 8px rgba(255, 255, 255, 0.3);
}

/* ปรับแต่งกล่องข้อความแชทให้ดูมีมิติ */
[data-testid="stChatMessage"] {
     background-color: rgba(255, 255, 255, 0.05);
     border-radius: 15px;
     border: 1px solid rgba(138, 43, 226, 0.3); /* ขอบสีม่วงสว่าง */
     margin-bottom: 10px;
}

/* ช่องพิมพ์ข้อความด้านล่าง */
[data-testid="stChatInput"] textarea {
    background-color: #2E004E !important; /* พื้นหลังช่องพิมพ์สีม่วงเข้ม */
    color: white !important;
    border: 2px solid rgba(138, 43, 226, 0.5) !important;
    border-radius: 20px;
    box-shadow: 0 0 15px rgba(138, 43, 226, 0.3) !important; /* ออร่าสีม่วงรอบช่องพิมพ์ */
}

/* ปุ่มใน Sidebar */
.stButton>button {
    background-color: #4B0082; /* สีปุ่มม่วงเข้ม */
    color: white;
    border: 1px solid rgba(255, 255, 255, 0.4);
    border-radius: 12px;
    box-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
    transition: all 0.3s ease;
}

/* เอฟเฟกต์ตอนเอาเมาส์ชี้ปุ่ม */
.stButton>button:hover {
     background-color: #6A0DAD;
     border-color: white;
     box-shadow: 0 0 20px rgba(255, 255, 255, 0.6); /* ออร่าขาววิ้งๆ */
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# --- User Interface ---
def clear_history():
    st.session_state["messages"] = [
        {"role": "model", "content": "สวัสดีค่ะ น้อง Graphic Bot สายดาร์ก(แต่น่ารัก) พร้อมให้บริการความรู้เรื่องกราฟิกแล้วค่า 🔮✨"}
    ]
    st.rerun()

with st.sidebar:
    st.title("เมนูคำสั่ง ⚡")
    if st.button("🗑️ ล้างประวัติการคุย"):
        clear_history()
    st.markdown("---")
    st.write("Theme: Dark Magic 🔮")

# ปรับ Title ให้เข้ากับตีมใหม่
st.title("✨🔮 น้อง Graphic Bot: Dark Aura Edition 🎨🌌")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "model", "content": "สวัสดีค่ะ น้อง Graphic Bot สายดาร์ก(แต่น่ารัก) พร้อมให้บริการความรู้เรื่องกราฟิกแล้วค่า 🔮✨"}
    ]

# แสดงประวัติ (ใช้อีโมจิเดิมที่น่ารักเข้ากับตีมได้ดี)
for msg in st.session_state["messages"]:
    # 🐰 User, 🦄 Bot (ยูนิคอร์นเข้ากับตีมเวทมนตร์ได้ดี)
    avatar_icon = "🐰" if msg["role"] == "user" else "🦄"
    st.chat_message(msg["role"], avatar=avatar_icon).write(msg["content"])

# รับ Input
if prompt := st.chat_input("พิมพ์คำถามเกี่ยวกับกราฟิกที่นี่เลยค่า... ✨"):
    st.session_state["messages"].append({"role": "user", "content": prompt})
    st.chat_message("user", avatar="🐰").write(prompt)

    def generate_response():
        # สร้าง History
        history_api = [
            {"role": msg["role"], "parts": [{"text": msg["content"]}]}
            for msg in st.session_state["messages"]
        ]

        try:
            # ตรวจสอบว่า model รองรับ start_chat หรือไม่ (gemini-pro บางรุ่นใช้ generate_content แทน)
            if hasattr(model, 'start_chat'):
                chat_session = model.start_chat(history=history_api)
            else:
                # กรณีใช้โมเดลรุ่นเก่าที่ไม่รองรับ start_chat โดยตรง (Fallback)
                chat_session = model

            # --- 🔥 จุดแก้เผ็ด AI มั่ว: บังคับคำสั่งแนบท้ายทุกครั้ง (Suffix Prompting) 🔥 ---
            strict_prompt = f"""
            {prompt}

            (IMPORTANT COMMAND FOR AI:
            1. Answer purely based on the provided CONTEXT above in System Instructions.
            2. If the answer is NOT in the CONTEXT, you MUST say "ขออภัยค่ะ ไม่มีข้อมูลเรื่องนี้ในเอกสารค่ะ 🥺"
            3. DO NOT use outside knowledge to answer.
            4. Keep the tone cute and cheerful as instructed.)
            """

            # ส่งข้อความ (รองรับทั้งสองแบบ)
            if hasattr(chat_session, 'send_message'):
                 response = chat_session.send_message(strict_prompt)
            else:
                 # สำหรับโมเดลที่ไม่มี chat session (อาจจะไม่รองรับ history แบบต่อเนื่องดีนัก)
                 full_prompt_for_generate = FULL_SYSTEM_INSTRUCTION + "\n\nHistory:\n" + str(history_api) + "\n\nUser Input:\n" + strict_prompt
                 response = chat_session.generate_content(full_prompt_for_generate)


            st.session_state["messages"].append({"role": "model", "content": response.text})
            st.chat_message("model", avatar="🦄").write(response.text)

        except Exception as e:
            st.error(f"ระบบขัดข้องชั่วคราว: {e}")

    with st.spinner('กำลังร่ายมนตร์ค้นหาคำตอบ... 🔮💫'):
        generate_response()
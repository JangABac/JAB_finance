# python.py

import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

# Tiêu đề in đậm, màu đỏ, cuối có biểu tượng $
st.markdown("<h1 style='color:red; font-weight:bold;'>Ứng dụng Phân Tích Báo Cáo Tài Chính $</h1>", unsafe_allow_html=True)

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini (phân tích tài chính) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            try:
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Chỉ số Thanh toán Hiện hành (Năm trước)", f"{thanh_toan_hien_hanh_N_1:.2f} lần")
                with col2:
                    st.metric("Chỉ số Thanh toán Hiện hành (Năm sau)", f"{thanh_toan_hien_hanh_N:.2f} lần",
                              delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}")
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN'.")

            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%", 
                    "N/A", 
                    "N/A"
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API.")
    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc file: {e}")
else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

# ======================================================================
# 💬 KHUNG CHAT VỚI GEMINI
# ======================================================================

st.markdown("---")
st.subheader("💬 Khung Chat với Gemini")

with st.expander("⚙️ Cấu hình Chat (tuỳ chọn)", expanded=False):
    model = st.selectbox("Chọn model",
                         ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-2.5-pro"],
                         index=0)
    system_prompt = st.text_area("System prompt",
        value="Bạn là trợ lý Python & tài chính. Trả lời ngắn gọn, có ví dụ khi cần.", height=100)
    max_history = st.slider("Số lượt hội thoại đưa vào ngữ cảnh", 2, 20, 12)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for msg in st.session_state.chat_messages:
    with st.chat_message("user" if msg["role"] == "user" else "assistant"):
        st.markdown(msg["content"])

def _format_history_for_prompt(history, system_prompt, last_user):
    lines = [f"[SYSTEM]\n{system_prompt.strip()}", ""]
    for turn in history[-max_history:]:
        if turn["role"] == "user":
            lines.append(f"[USER]\n{turn['content']}\n")
        else:
            lines.append(f"[ASSISTANT]\n{turn['content']}\n")
    lines.append(f"[USER]\n{last_user}\n")
    lines.append("[ASSISTANT]\n")
    return "\n".join(lines).strip()

def _call_gemini_chat(api_key, model_name, system_prompt, history, user_text):
    try:
        client = genai.Client(api_key=api_key)
        prompt_text = _format_history_for_prompt(history, system_prompt, user_text)
        resp = client.models.generate_content(model=model_name, contents=prompt_text)
        return (resp.text or "").strip()
    except Exception as e:
        return f"⚠️ Lỗi gọi Gemini: {e}"

user_input = st.chat_input("Nhập câu hỏi cho Gemini...")

if user_input:
    st.session_state.chat_messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        assistant_reply = "⚠️ Chưa cấu hình `GEMINI_API_KEY`."
        with st.chat_message("assistant"):
            st.markdown(assistant_reply)
        st.session_state.chat_messages.append({"role": "assistant", "content": assistant_reply})
    else:
        with st.chat_message("assistant"):
            with st.spinner("Gemini đang soạn trả lời..."):
                assistant_reply = _call_gemini_chat(api_key, model, system_prompt,
                                                    st.session_state.chat_messages[:-1], user_input)
                st.markdown(assistant_reply)
        st.session_state.chat_messages.append({"role": "assistant", "content": assistant_reply})

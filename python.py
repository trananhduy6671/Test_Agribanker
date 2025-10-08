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

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # ******************************* PHẦN SỬA LỖI BẮT ĐẦU *******************************
    # Lỗi xảy ra khi dùng .replace() trên giá trị đơn lẻ (numpy.int64).
    # Sử dụng điều kiện ternary để xử lý giá trị 0 thủ công cho mẫu số.
    
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    # ******************************* PHẦN SỬA LỖI KẾT THÚC *******************************
    
    return df

# --- Hàm gọi API Gemini ---
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
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
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
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            try:
                # Lọc giá trị cho Chỉ số Thanh toán Hiện hành (Ví dụ)
                
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn (Dùng giá trị giả định hoặc lọc từ file nếu có)
                # **LƯU Ý: Thay thế logic sau nếu bạn có Nợ Ngắn Hạn trong file**
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    )
                    
            except IndexError:
                 st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
                 thanh_toan_hien_hanh_N = "N/A" # Dùng để tránh lỗi ở Chức năng 5
                 thanh_toan_hien_hanh_N_1 = "N/A"
            
            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI
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
                    f"{thanh_toan_hien_hanh_N_1}", 
                    f"{thanh_toan_hien_hanh_N}"
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
                     st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")
# ========================== KHUNG CHAT GEMINI (BỔ SUNG) ==========================
# Bạn có thể đặt block này ở phía trên hoặc dưới cùng file. Mặc định mình đặt sau phần tính toán.
with st.sidebar:
    st.markdown("## 💬 Chat với Gemini")
    st.caption("Hỏi đáp nhanh về số liệu, công thức, hoặc giải thích kết quả. Lịch sử chat chỉ tồn tại trong phiên hiện tại.")

# Khởi tạo lịch sử chat trong session_state
if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = [
        {"role": "system", "content": (
            "Bạn là một trợ lý tài chính nói tiếng Việt, am hiểu kế toán – kiểm toán – phân tích tài chính. "
            "Giải thích ngắn gọn, đưa ra công thức khi phù hợp, có thể trích số liệu từ bảng nếu người dùng cung cấp."
        )}
    ]

def _to_genai_contents(history: list, user_message: str):
    """
    Chuyển lịch sử hội thoại + câu hỏi mới về format contents cho Gemini.
    'history' là list các dict: {'role': 'user'/'assistant'/'system', 'content': str}
    """
    contents = []
    for msg in history:
        # Map role sang định dạng Gemini
        role = "user" if msg["role"] == "user" else ("model" if msg["role"] == "assistant" else "user")
        # System message đưa như user prompt mở đầu
        contents.append({"role": role, "parts": [msg["content"]]})
    # Thêm câu hỏi mới
    contents.append({"role": "user", "parts": [user_message]})
    return contents

def gemini_chat_reply(history: list, user_message: str, api_key: str) -> str:
    """
    Gửi hội thoại (history) + câu hỏi mới đến Gemini và trả về câu trả lời dạng text.
    """
    try:
        client = genai.Client(api_key=api_key)
        model_name = "gemini-2.5-flash"
        contents = _to_genai_contents(history, user_message)
        resp = client.models.generate_content(model=model_name, contents=contents)
        return getattr(resp, "text", "").strip() or "Mình chưa nhận được nội dung phản hồi từ Gemini."
    except APIError as e:
        return f"Lỗi gọi Gemini API (chat): {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi khi chat với Gemini: {e}"

# KHU VỰC HIỂN THỊ CHAT (ở trang chính, không phải sidebar)
st.divider()
st.subheader("💬 Khung Chat hỏi đáp với Gemini")

# Hiển thị lịch sử tin nhắn (bỏ qua 'system' để giao diện gọn hơn)
for msg in st.session_state.chat_messages:
    if msg["role"] == "system":
        continue
    with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
        st.markdown(msg["content"])

# Ô nhập chat
user_input = st.chat_input("Nhập câu hỏi (ví dụ: 'Giải thích cách tính tỷ trọng năm sau?')")
if user_input:
    st.session_state.chat_messages.append({"role": "user", "content": user_input})

    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        with st.chat_message("assistant"):
            st.error("Chưa cấu hình `GEMINI_API_KEY` trong Secrets.")
    else:
        # Nếu người dùng đã tải file và có df_processed, bạn có thể cho phép tham chiếu nhanh
        context_hint = ""
        if "df_processed" in locals() and isinstance(df_processed, pd.DataFrame):
            try:
                # Rút gọn bối cảnh để chat thông minh hơn (tránh quá dài)
                head_preview = df_processed.head(10).to_markdown(index=False)
                context_hint = (
                    "\n\n[Ngữ cảnh dữ liệu (preview 10 dòng đầu)]:\n" + head_preview +
                    "\n\nHãy ưu tiên trích dẫn/giải thích dựa trên các cột: "
                    "'Chỉ tiêu', 'Năm trước', 'Năm sau', 'Tốc độ tăng trưởng (%)', "
                    "'Tỷ trọng Năm trước (%)', 'Tỷ trọng Năm sau (%)'."
                )
            except Exception:
                context_hint = ""

        # Gửi và nhận phản hồi
        with st.chat_message("assistant"):
            with st.spinner("Gemini đang trả lời..."):
                reply = gemini_chat_reply(
                    history=st.session_state.chat_messages,
                    user_message=(user_input + context_hint),
                    api_key=api_key
                )
                st.markdown(reply)
        # Lưu phản hồi vào lịch sử
        st.session_state.chat_messages.append({"role": "assistant", "content": reply})
# ======================== HẾT KHUNG CHAT GEMINI (BỔ SUNG) ========================

import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.api_core import exceptions

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính với Chatbot AI 📊🤖")

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

# --- Hàm gọi API Gemini cho Phân tích Tổng quan ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét tổng quan."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = model.generate_content(prompt)
        return response.text

    except exceptions.GoogleAPICallError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra lại Khóa API. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"

# --- Hàm khởi tạo mô hình Chat ---
def initialize_chat_model(api_key, data_context):
    """Khởi tạo mô hình Gemini và bắt đầu một phiên trò chuyện với context."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        system_instruction = f"""
        Bạn là một trợ lý AI chuyên về phân tích tài chính. Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng một cách ngắn gọn, chính xác. 
        Mọi câu trả lời PHẢI dựa hoàn toàn vào dữ liệu từ báo cáo tài chính được cung cấp dưới đây. 
        Nếu câu hỏi không thể trả lời được từ dữ liệu, hãy trả lời là 'Tôi không có đủ thông tin từ dữ liệu được cung cấp để trả lời câu hỏi này.'
        
        **Dữ liệu Báo cáo Tài chính:**
        {data_context}
        """
        
        chat = model.start_chat(history=[
            {'role': 'user', 'parts': [system_instruction]},
            {'role': 'model', 'parts': ["Tôi đã sẵn sàng. Vui lòng đặt câu hỏi của bạn về dữ liệu tài chính đã cung cấp."]}
        ])
        return chat
    except Exception:
        return None

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

        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = None

        st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
        st.dataframe(df_processed.style.format({
            'Năm trước': '{:,.0f}', 'Năm sau': '{:,.0f}',
            'Tốc độ tăng trưởng (%)': '{:.2f}%',
            'Tỷ trọng Năm trước (%)': '{:.2f}%', 'Tỷ trọng Năm sau (%)': '{:.2f}%'
        }), use_container_width=True)
        
        st.subheader("4. Các Chỉ số Tài chính Cơ bản")
        try:
            tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', na=False)]['Năm sau'].iloc[0]
            tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', na=False)]['Năm trước'].iloc[0]
            no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', na=False)]['Năm sau'].iloc[0]
            no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', na=False)]['Năm trước'].iloc[0]
            thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else 0
            thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else 0
            
            col1, col2 = st.columns(2)
            col1.metric("Chỉ số Thanh toán Hiện hành (Năm trước)", f"{thanh_toan_hien_hanh_N_1:.2f} lần")
            col2.metric("Chỉ số Thanh toán Hiện hành (Năm sau)", f"{thanh_toan_hien_hanh_N:.2f} lần", f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}")
        except (IndexError, KeyError):
             st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
             thanh_toan_hien_hanh_N, thanh_toan_hien_hanh_N_1 = "N/A", "N/A"
        
        st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
        data_for_ai = df_processed.to_markdown(index=False)
        if st.button("Yêu cầu AI Phân tích Tổng quan"):
            api_key = st.secrets.get("GEMINI_API_KEY")
            if api_key:
                with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                    ai_result = get_ai_analysis(data_for_ai, api_key)
                    st.info(ai_result)
            else:
                 st.error("Lỗi: Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

        # --- PHẦN MỚI: KHUNG CHAT VỚI AI ---
        st.subheader("6. Trò chuyện với AI về Báo cáo Tài chính")
        
        # Lấy API Key và khởi tạo session chat nếu chưa có
        api_key = st.secrets.get("GEMINI_API_KEY")
        if not api_key:
            st.warning("Vui lòng thêm GEMINI_API_KEY vào mục Secrets để kích hoạt tính năng trò chuyện.")
        elif st.session_state.chat_session is None:
            with st.spinner("Đang khởi tạo Trợ lý AI..."):
                data_context_for_chat = df_processed.to_markdown(index=False)
                st.session_state.chat_session = initialize_chat_model(api_key, data_context_for_chat)
                if st.session_state.chat_session is None:
                    st.error("Khởi tạo Trợ lý AI thất bại. Vui lòng kiểm tra lại API Key.")
                else:
                    # Xóa lịch sử cũ và thêm tin nhắn chào mừng
                    st.session_state.messages = [{"role": "assistant", "content": "Tôi có thể giúp gì cho bạn về dữ liệu này?"}]
        
        # Hiển thị lịch sử trò chuyện
        if st.session_state.chat_session:
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Nhận input từ người dùng
            if prompt := st.chat_input("Hãy đặt câu hỏi về các chỉ số tài chính..."):
                # Thêm tin nhắn của người dùng vào lịch sử và hiển thị
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                # Gửi tin nhắn đến Gemini và nhận câu trả lời
                with st.chat_message("assistant"):
                    with st.spinner("AI đang suy nghĩ..."):
                        try:
                            response = st.session_state.chat_session.send_message(prompt)
                            response_text = response.text
                        except Exception as e:
                            response_text = f"Đã xảy ra lỗi: {e}"
                        
                        st.markdown(response_text)
                
                # Thêm câu trả lời của AI vào lịch sử
                st.session_state.messages.append({"role": "assistant", "content": response_text})

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")
else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

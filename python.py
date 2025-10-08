import streamlit as st
import pandas as pd
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPICallError

# --- Cấu hình Trang Streamlit và API Key ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài Chính 📊")

# Lấy API key từ Streamlit Secrets và cấu hình
try:
    api_key = st.secrets.get("GEMINI_API_KEY")
    if api_key:
        genai.configure(api_key=api_key)
    else:
        st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")
except Exception as e:
    st.error(f"Lỗi khi cấu hình Gemini API: {e}")


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

# --- Hàm gọi API Gemini cho phân tích ban đầu ---
def get_initial_ai_analysis(data_for_ai):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét ban đầu."""
    try:
        # THAY ĐỔI: Chuyển sang model 'gemini-pro' để ổn định hơn
        model = genai.GenerativeModel('gemini-pro')
        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """
        response = model.generate_content(prompt)
        return response.text
    except GoogleAPICallError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
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
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else 0
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else 0

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
            except (IndexError, ZeroDivisionError):
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' hoặc mẫu số bằng 0 để tính chỉ số.")
                thanh_toan_hien_hanh_N = "N/A"
                thanh_toan_hien_hanh_N_1 = "N/A"

            # --- Chức năng 5: Nhận xét AI ---
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
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if not isinstance(thanh_toan_hien_hanh_N, str) else "N/A",
                    f"{thanh_toan_hien_hanh_N_1}",
                    f"{thanh_toan_hien_hanh_N}"
                ]
            }).to_markdown(index=False)

            if st.button("Yêu cầu AI Phân tích"):
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_initial_ai_analysis(data_for_ai)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Không thể phân tích. Vui lòng kiểm tra lại cấu hình API Key.")

            # ******************************* PHẦN CODE MỚI BẮT ĐẦU *******************************
            # --- Chức năng 6: Trò chuyện và Phân tích sâu hơn với AI ---
            st.subheader("6. Trò chuyện và Phân tích sâu hơn với AI")

            # Khởi tạo model chat và lịch sử trò chuyện trong session state
            if "chat_model" not in st.session_state:
                # THAY ĐỔI: Chuyển sang model 'gemini-pro' để ổn định hơn
                st.session_state.chat_model = genai.GenerativeModel('gemini-pro')
            if "chat_session" not in st.session_state:
                 # Bắt đầu cuộc trò chuyện với một context ban đầu về dữ liệu
                initial_prompt = f"""
                Bối cảnh: Tôi đã cung cấp cho bạn dữ liệu tài chính của một công ty. Dữ liệu này bao gồm bảng phân tích chi tiết.
                
                Dữ liệu đây:
                {data_for_ai}
                
                Nhiệm vụ của bạn: Hãy đóng vai một chuyên gia phân tích tài chính. Trả lời các câu hỏi của tôi một cách ngắn gọn, chuyên nghiệp và chỉ dựa vào dữ liệu đã được cung cấp. Bắt đầu bằng cách xác nhận bạn đã sẵn sàng.
                """
                st.session_state.chat_session = st.session_state.chat_model.start_chat(history=[
                    {"role": "user", "parts": [initial_prompt]},
                    {"role": "model", "parts": ["Tôi đã sẵn sàng. Tôi đã xem xét dữ liệu tài chính bạn cung cấp. Bạn muốn hỏi điều gì cụ thể?"]}
                ])

            # Hiển thị các tin nhắn đã có
            for message in st.session_state.chat_session.history[1:]: # Bỏ qua prompt context ban đầu
                with st.chat_message(message.role):
                    st.markdown(message.parts[0].text)

            # Nhận input từ người dùng
            if prompt := st.chat_input("Bạn muốn hỏi gì về dữ liệu này?"):
                if not api_key:
                    st.error("Không thể chat. Vui lòng kiểm tra lại cấu hình API Key.")
                else:
                    # Hiển thị tin nhắn của người dùng
                    with st.chat_message("user"):
                        st.markdown(prompt)

                    # Gửi tin nhắn đến Gemini và nhận phản hồi
                    try:
                        with st.spinner("Gemini đang suy nghĩ..."):
                            response = st.session_state.chat_session.send_message(prompt)
                        # Hiển thị phản hồi của AI
                        with st.chat_message("model"):
                            st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Đã có lỗi xảy ra: {e}")
            # ******************************* PHẦN CODE MỚI KẾT THÚC *******************************

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")


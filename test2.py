import streamlit as st
import qrcode
import numpy as np
import cv2
from PIL import Image
from pyzbar.pyzbar import decode
import io

# Page Config
st.set_page_config(layout="centered", page_title="QR Code Generator & Scanner")

# CSS Styling
st.markdown("""
    <style>
        .title {text-align: center; font-size: 36px; font-weight: bold; margin-bottom: 20px;}
        .box {border: 2px solid #ddd; padding: 20px; border-radius: 10px; margin-bottom: 20px;}
        .result {background-color: #e0f7fa; padding: 10px; border-radius: 5px; margin-top: 10px;}
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="title">🔳 QR Code Generator & Scanner</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

# --- QR Code Generator ---
with col1:
    st.markdown('<div class="box"><h2>Generate QR Code</h2></div>', unsafe_allow_html=True)
    text_input = st.text_input("Enter text or URL to generate QR code:")
    generate_btn = st.button("Generate")

    if generate_btn:
        if text_input.strip():
            qr = qrcode.QRCode(
                version=None,
                error_correction=qrcode.constants.ERROR_CORRECT_M,
                box_size=10,
                border=4,
            )
            qr.add_data(text_input)
            qr.make(fit=True)
            img = qr.make_image(fill_color="black", back_color="white").convert('RGB')

            st.image(img, caption="Generated QR Code")

            # Download button
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            byte_im = buffer.getvalue()

            st.download_button(
                label="📥 Download QR Code",
                data=byte_im,
                file_name="qr_code.png",
                mime="image/png"
            )
        else:
            st.warning("Please enter text to generate a QR code.")

# --- QR Code Detector from Image ---
with col2:
    st.markdown('<div class="box"><h2>Detect QR Code (from image)</h2></div>', unsafe_allow_html=True)
    uploaded_qr = st.file_uploader("Upload a QR code image:", type=["png", "jpg", "jpeg"])

    if st.button("Detect from Image"):
        if uploaded_qr:
            try:
                img = Image.open(uploaded_qr).convert("RGB")
                img_np = np.array(img)

                # Resize large images
                max_dim = 800
                if img_np.shape[0] > max_dim or img_np.shape[1] > max_dim:
                    img_np = cv2.resize(img_np, (max_dim, max_dim), interpolation=cv2.INTER_AREA)

                detector = cv2.QRCodeDetector()
                data, bbox, _ = detector.detectAndDecode(img_np)

                if data:
                    st.markdown(f'<div class="result">✅ QR Code Data: <strong>{data}</strong></div>', unsafe_allow_html=True)
                    if bbox is not None:
                        points = np.int32(bbox).reshape(-1, 2)
                        for i in range(len(points)):
                            pt1 = tuple(points[i])
                            pt2 = tuple(points[(i + 1) % len(points)])
                            cv2.line(img_np, pt1, pt2, (0, 255, 0), 2)
                    st.image(img_np, caption="Detected QR Code with Bounding Box")
                else:
                    # Try fallback using pyzbar
                    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                    qr_codes = decode(gray)
                    if qr_codes:
                        for qr in qr_codes:
                            data = qr.data.decode("utf-8")
                            st.markdown(f'<div class="result">✅ QR Code Data: <strong>{data}</strong></div>', unsafe_allow_html=True)
                            (x, y, w, h) = qr.rect
                            cv2.rectangle(img_np, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        st.image(img_np, caption="Detected QR Code with Bounding Box")
                    else:
                        st.warning("No QR code detected. Try a clearer or higher resolution image.")
            except Exception as e:
                st.error(f"Error: {str(e)}. Please upload a valid image.")
        else:
            st.error("Please upload a QR code image.")

# --- QR Code Detection via Webcam ---
st.markdown('<div class="box"><h2>📷 Live QR Code Scanner (Webcam)</h2></div>', unsafe_allow_html=True)
run_camera = st.checkbox("Start Camera Scanner")

if run_camera:
    st.info("Press 'Stop' checkbox to end scanning.")
    frame_window = st.image([])

    cap = cv2.VideoCapture(0)
    detector = cv2.QRCodeDetector()

    scanned = False
    while run_camera and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access webcam. Make sure it's not being used by another app.")
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        data, bbox, _ = detector.detectAndDecode(frame_rgb)

        if bbox is not None and data:
            points = np.int32(bbox).reshape(-1, 2)
            for i in range(len(points)):
                pt1 = tuple(points[i])
                pt2 = tuple(points[(i + 1) % len(points)])
                cv2.line(frame_rgb, pt1, pt2, (0, 255, 0), 2)

            cv2.putText(frame_rgb, data, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            st.success(f"✅ QR Code Scanned: {data}")
            scanned = True
            frame_window.image(frame_rgb)
            break

        frame_window.image(frame_rgb)

    cap.release()

# Base image nhẹ gọn
FROM python:3.11-slim

# Tạo thư mục app bên trong container
WORKDIR /app

# Copy toàn bộ mã nguồn và dữ liệu vào container
COPY . /app

# Cài đặt thư viện cần thiết
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000 (API FastAPI sẽ chạy ở đây)
EXPOSE 8000

# Câu lệnh mặc định khi container khởi động
CMD ["uvicorn", "src.api_server:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

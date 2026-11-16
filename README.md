# Hướng dẫn chạy phần triển khai Day 7

Thư mục này chứa các thử nghiệm truy xuất và bộ chuyển tiếp agent dùng mô hình thật. Bộ dữ liệu cùng benchmark nằm trong `../data/`; trace và báo cáo kết quả được ghi vào `results/`.

## Tạo lại dữ liệu từ Day 6

Chạy từ thư mục gốc của repo Day 7:

```powershell
python submission/build_dataset.py --day6-root C:\AI_K20\Lab\K5-L3-Day06-2A202603731-DataIngestionEngineering
```

Lệnh này dùng lại bộ chia đoạn theo đoạn văn của Day 6, sao chép 37 đơn vị truyện đã xử lý, xác minh bằng chứng của mọi câu có đáp án với văn bản nguồn và chunk, rồi ghi benchmark 24 câu. Các trường đáp án chuẩn không được gửi vào retriever hoặc agent.

## Cài các thư viện cần thiết

```powershell
python -m venv submission/.venv
.\submission\.venv\Scripts\python.exe -m pip install -r submission/requirements.txt
```

Collection Chroma của Day 6 cần còn ở `submission/starter/results/chroma_controlled` trong thư mục Day 6; đồng thời cần cài được model Sentence Transformers tương ứng. `run_retrieval.py` mở collection Day 6 hiện có, không tạo lại collection.

## Thử nghiệm truy xuất

Chạy baseline TF-IDF và BM25 offline:

```powershell
.\submission\.venv\Scripts\python.exe submission/run_lexical.py --split dev --output submission/results/lexical-dev
.\submission\.venv\Scripts\python.exe submission/run_lexical.py --split dev --top-k 5 --budget-chars 4000 --output submission/results/bm25-top5-dev
```

Dùng `run_hybrid_cached.py` để so sánh có kiểm soát trên dev. Script này kết hợp thứ hạng BM25 mới với thứ hạng dense B đã lưu từ Day 6 cho cùng 15 câu hỏi dev; script chỉ chạy trên dev.

Sau khi chọn cấu hình bằng dev, chạy test mới:

```powershell
.\submission\.venv\Scripts\python.exe submission/run_retrieval.py --split test --top-k 5 --budget-chars 4000 --output submission/results/retrieval-test
```

## Chạy agent bằng mô hình thật

Với DeepSeek, đặt `DEEPSEEK_API_KEY` trong file `.env` cục bộ. Bộ chuyển tiếp dùng `deepseek-flash` và `https://api.deepseek.com` theo mặc định; có thể đổi bằng `DAY7_MODEL` hoặc `DAY7_BASE_URL`. Agent tắt chế độ thinking cho vòng gọi tool cơ bản để không cần gửi lại trường suy luận riêng của provider. Có hỗ trợ OpenAI và OpenRouter. Xem [hướng dẫn gọi tool chính thức của DeepSeek](https://api-docs.deepseek.com/guides/tool_calls/). Không dán khóa vào báo cáo hoặc commit lên repo. Runner đọc `.env` ở thư mục repo, `submission/` hoặc thư mục Lab cha.

```powershell
.\submission\.venv\Scripts\python.exe submission/run_agent.py --split dev --output submission/results/agent-dev
.\submission\.venv\Scripts\python.exe submission/run_agent.py --split test --output submission/results/agent-test
```

Để dùng retriever BM25 đã chọn thay cho dense Day 6 mặc định, thêm `--retrieval bm25 --top-k 5 --budget 4000`. Để chạy baseline TF-IDF của agent, dùng `--retrieval tfidf --top-k 3 --budget 2400`. Cả hai cấu hình đều gọi DeepSeek.

Agent chỉ nhận câu hỏi, các đoạn đã truy xuất và ngữ cảnh tìm kiếm trước đó. Đáp án chuẩn chỉ được dùng bởi bộ đánh giá offline. Hãy rà citation với văn bản nguồn, sau đó dùng `starter/agent_metrics.py` cùng nhãn review. `summarize_agent_usage.py` ước tính chi phí dựa trên usage do API trả về và mức giá off-peak được ghi trong kết quả.

## Giới hạn đã biết

- Agent DeepSeek thật đã chạy với TF-IDF và BM25. Bộ chuyển tiếp dense đã có, nhưng lượt chạy dense mới trên test cần cài Chroma và Sentence Transformers trong venv.
- Khóa API đã dùng trong phiên trước không được lưu trong repo; hãy dùng khóa đã xoay trong file `.env` bị Git bỏ qua.
- So sánh dense dev dùng lại trace collection B của Day 6 và ghi rõ nguồn gốc; không tuyên bố đây là lượt chạy dense mới của Day 7.
- Benchmark cơ bản dùng dữ kiện văn học tiếng Anh. Các câu test có đáp án được viết riêng cho Day 7 và evidence đã được kiểm tra với corpus.
- Project Gutenberg nêu tình trạng public domain tại Hoa Kỳ; hãy kiểm tra quy định tái phân phối ở địa phương và chính sách tải corpus lên dịch vụ bên ngoài trước khi chia sẻ.

## Thử thách nâng cao: World Development Report 2021

Nguồn, thông tin ghi công, giấy phép, số trang, SHA-256 của PDF, sổ trang, phương pháp trích xuất, thống kê index và giới hạn xử lý được ghi trong `../data/advanced/source_manifest.json`. Báo cáo 350 trang có tại URL World Bank chính thức trong manifest và mang giấy phép CC BY 3.0 IGO; hãy kiểm tra ghi chú về nội dung bên thứ ba trước khi tái sử dụng từng hình riêng lẻ. PDF gốc và ảnh trang dùng để kiểm tra được Git bỏ qua. Tải PDF về máy trước khi dựng lại index:

```powershell
New-Item -ItemType Directory -Force data/advanced | Out-Null
Invoke-WebRequest -Uri 'https://documents1.worldbank.org/curated/en/248201616598597113/pdf/World-Development-Report-2021-Data-for-Better-Lives.pdf' -OutFile data/advanced/world_development_report_2021.pdf
python submission/advanced_index.py
python submission/build_advanced_benchmark.py
```

Index dùng Poppler `pdftotext -layout` trên toàn bộ trang, chia chunk theo trang và đoạn văn; retriever BM25 ưu tiên locator Figure/Map/Table có trong truy vấn. Caption, text vector, nhãn và nội dung bảng được lập chỉ mục; citation trong câu trả lời trỏ tới chunk ID, phần đầu context của chunk ghi số trang PDF vật lý. Với câu hỏi trực quan, `data/advanced/benchmark.json` lưu loại hình và trang nguồn để tra cứu. Các trang được kiểm tra trực quan nằm cục bộ trong `data/advanced/evidence_pages/`. Có 15 trang không có text máy đọc được và chưa OCR; danh sách nằm trong page ledger. Trích xuất text không thay thế việc kiểm tra hình gốc.

Split cố định trong `data/advanced/benchmark.json`: dev 18 câu, test 30 câu. Lượt test cuối dùng DeepSeek `deepseek-flash`, BM25 có ưu tiên locator, top-k 8, context 18.000 ký tự và tối đa 3 lần tìm kiếm. `submission/results/advanced_locked_config.json` lưu cấu hình cùng hash input trước khi chạy test. Đặt `DEEPSEEK_API_KEY` và `DAY7_MODEL=deepseek-flash` trong `.env` cục bộ ở repo hoặc thư mục Lab cha; tuyệt đối không commit khóa API.

```powershell
python submission/run_agent.py --questions data/advanced/benchmark.json --chunks data/advanced/chunks.json --split dev --retrieval bm25 --model deepseek-flash --top-k 8 --budget 18000 --max-searches 3 --output submission/results/advanced-bm25-dev-k8
python submission/run_agent.py --questions data/advanced/benchmark.json --chunks data/advanced/chunks.json --split test --retrieval bm25 --model deepseek-flash --top-k 8 --budget 18000 --max-searches 3 --output submission/results/advanced-bm25-test-final
python submission/evaluate_advanced_test.py
```

Test đạt **29/30 (96,67%)** theo nhãn được rà thủ công. Metrics theo nhóm, citation từng câu, trace, thời gian, token, chi phí ước tính và lý do chấm nằm trong `submission/results/advanced-bm25-test-final/`. Câu chưa đạt duy nhất là WDR-T19: câu có evidence nhưng agent từ chối. Chi phí được ước tính từ usage API và mức giá off-peak; hóa đơn thực tế của provider có thể khác.

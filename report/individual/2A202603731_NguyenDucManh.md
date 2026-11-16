# Báo cáo Lab Day 7 — Agentic RAG

## Dữ liệu và benchmark

- **Nguồn:** 37 truyện Sherlock Holmes từ Project Gutenberg #1661, #834 và #108; 37 đơn vị truyện, không phân trang vật lý. Nội dung thuộc phạm vi công cộng theo Project Gutenberg tại Hoa Kỳ; trước khi phân phối hoặc tải nội dung lên dịch vụ bên ngoài cần xác minh quy định tại địa phương.
- **Chunking:** paragraph-aware, tối đa 1.000 ký tự, cùng nội dung và ID xác định với Day 6 collection B. Đầu ra gồm 2.112 chunk, có `chunk_id`, tuyển tập, truyện và metadata nguồn.
- **Vector store:** Chroma PersistentClient, collection `day6_sherlock_b`, cosine/HNSW, Chroma 1.5.9. Dense query dùng `sentence-transformers/all-MiniLM-L6-v2`, 384 chiều, cùng model tạo document/query embedding.
- **Benchmark:** 24 câu: dev 15 có evidence + 2 thiếu evidence; test mới 5 có evidence + 2 thiếu evidence. Năm câu có đáp án trong test được tạo mới cho Day 7 vì test Day 6 đã được dùng trong lượt đánh giá Day 6. Evidence được kiểm tra ở đúng nguồn, truyện và chunk. Câu thiếu evidence được xác nhận bằng kiểm tra toàn bộ 37 văn bản cho cụm liên quan.
- **Agent model:** DeepSeek `deepseek-flash`, dùng function tool calling. Agent đã chạy thật với BM25 và TF-IDF trên dev/test. Dense adapter cho agent có sẵn nhưng chưa chạy do venv chưa cài Chroma/Sentence Transformers; dense dev trong bảng là trace Day 6 đã lưu.

## Retrieval và thử nghiệm

Tất cả số dưới đây dùng cùng 15 câu dev có đáp án. Hit@1 và Evidence Hit@3 yêu cầu khớp evidence ở đúng nguồn và truyện. Context Evidence đo evidence trong context được đóng gói. BM25/TF-IDF latency là median theo truy vấn, không gồm thời gian tạo index. Dense và hybrid dùng lại ranking dense B đã được lưu từ lượt chạy Day 6; per-query latency không có trong trace Day 6.

| Cấu hình | Phương pháp | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence | Thời gian / chi phí |
|---|---|---|---|---:|---:|---:|---|
| Baseline | TF-IDF, top-3, context 2.400 ký tự | Baseline lexical offline | dev | 6/15 (40%) | 9/15 (60%) | 9/15 (60%) | median 6.77 ms; $0 |
| Phương pháp 2 | Dense Day 6 collection B, top-3 | Dense retrieval paragraph chunks | dev | 7/15 (46.7%) | 8/15 (53.3%) | 7/15 (46.7%) | trace Day 6; không lưu latency từng câu; $0 |
| Phương pháp 3 | BM25, top-3, context 2.400 ký tự | Từ khóa có trọng số IDF; không cộng score khác thang | dev | 8/15 (53.3%) | 11/15 (73.3%) | 10/15 (66.7%) | median 7.68 ms; $0 |
| Cải thiện 1 | RRF, hợp nhất top-3 dense B và top-3 BM25 | Hợp nhất thứ hạng để giảm lệ thuộc một tín hiệu | dev | 7/15 (46.7%) | 10/15 (66.7%) | 9/15 (60%) | latency RRF riêng chưa đo; $0 |
| Cải thiện 2 / chọn | BM25, top-5, context 4.000 ký tự | Thêm candidate và tăng ngân sách context | dev | 8/15 (53.3%) | 11/15 (73.3%) | 12/15 (80%) | median 8.15 ms; $0 |
| Baseline | TF-IDF, top-5, context 4.000 ký tự | So sánh test cùng điều kiện đã chốt | test | 0/5 | 0/5 | 0/5 | median 7.68 ms; $0 |
| Cấu hình đã chọn | BM25, top-5, context 4.000 ký tự | Chốt trên dev trước khi chạy test | test | 0/5 | 1/5 (20%) | 1/5 (20%) | median 7.71 ms; $0 |

BM25 top-3 cải thiện ranking so với TF-IDF; tăng top-k/context từ 3/2.400 lên 5/4.000 giữ nguyên Hit@1 và Hit@3 nhưng nâng Context Evidence từ 10/15 lên 12/15. Vì vậy chọn cấu hình top-5 cho bước context. Kết quả test kém cho thấy cấu hình này chưa tổng quát tốt; test được giữ nguyên, không dùng để chỉnh.

### Phân tích lỗi dev

- **SH005 — Silver Blaze:** BM25 top-3 trả về đúng truyện nhưng bỏ qua đoạn hỏi về án mạng John Straker và ngựa mất tích; đoạn evidence nằm sau phần mở đầu và không chứa đủ các từ khóa chung trong câu hỏi. Tăng lên top-5 chưa đưa đoạn đó vào context. Giới hạn còn lại: query mơ hồ, ranking ưu tiên các chunk có nhiều từ về Holmes, Silver Blaze và mystery.
- **SH015 — Five Orange Pips:** BM25 trả các đoạn về vụ Openshaw thay vì thời tiết ở Baker Street. Cụm từ chủ đề “weather” không trùng trực tiếp cách diễn đạt “wind” và “rain” trong evidence. Top-5 vẫn không khắc phục lỗi này. Đây là lỗi paraphrase/query vocabulary; cần thử query expansion hoặc reranking trên dev ở lượt tiếp theo.

## Agent và đánh giá

- **Thiết kế đã cài:** model gọi `search`; tối đa 3 search; TF-IDF baseline top-3/context 2.400 ký tự; BM25 chọn trên dev top-5/context 4.000 ký tự. Context khử trùng chunk theo ID. Passage được đánh dấu là dữ liệu, không phải chỉ dẫn. Trace lưu query, kết quả, context, quyết định/câu trả lời thô, citation, lỗi, thời gian và usage token.

| Cấu hình / split | Tool use | Grounded answer 0/1/2, trung bình /2 | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---:|---|---:|---:|---:|---:|
| TF-IDF, dev | 17/17 | 4 / 0 / 11; 1.47 | 13/13 | 12/12 | 2/2 | 3/15 |
| BM25 top-5, dev | 17/17 | 1 / 2 / 12; 1.73 | 15/15 | 14/14 | 2/2 | 1/15 |
| TF-IDF, test | 7/7 | 4 / 0 / 1; 0.40 | 1/2 | 2/2 | 2/2 | 3/5 |
| BM25 top-5, test | 7/7 | 4 / 0 / 1; 0.40 | 1/2 | 2/2 | 2/2 | 3/5 |

Ở test, D7T002 được trả lời đúng là “swamp adder”, nhưng citation chỉ mô tả con rắn và không nêu tên loài; chấm grounded answer 0 và citation không hợp lệ. TF-IDF trả lời đúng, có nguồn hỗ trợ D7T004; BM25 trả lời đúng, có nguồn hỗ trợ D7T005. Cả hai cấu hình từ chối đúng 2/2 câu thiếu evidence và từ chối nhầm 3/5 câu có đáp án. Nhãn dưới đây là lần rà soát hỗ trợ bằng AI; người nộp cần xác nhận citation và nhãn cuối.

Phân bố search: TF-IDF dev có 16 câu gọi 1 lần và 1 câu gọi 2 lần; BM25 dev có 16 câu gọi 1 lần và 1 câu gọi 3 lần; TF-IDF test có 4 câu gọi 1 lần và 3 câu gọi 3 lần; BM25 test có 5 câu gọi 1 lần và 2 câu gọi 2 lần.

| Cấu hình / split | API calls | Token vào / ra | Thời gian tổng | Chi phí ước tính |
|---|---:|---:|---:|---:|
| TF-IDF, dev | 35 | 22.330 / 2.290 | 37.68 s | $0.00286 |
| BM25 top-5, dev | 36 | 31.724 / 2.075 | 35.94 s | $0.00279 |
| TF-IDF, test | 20 | 14.153 / 939 | 21.49 s | $0.00173 |
| BM25 top-5, test | 16 | 14.885 / 800 | 16.37 s | $0.00158 |

Chi phí ước tính của bốn lượt đánh giá cuối là **$0.00896**; tổng mọi lượt gọi gồm hai lượt thử nghiệm dev trước khi sửa trace/citation là **179 requests, 155.885 tokens và khoảng $0.01655**. Ước tính dùng giá off-peak DeepSeek tại ngày chạy, tính riêng cache hit/miss và output token; giá có thể thay đổi theo giờ và bảng giá provider. Xem [bảng giá DeepSeek](https://api-docs.deepseek.com/quick_start/pricing/).

- **Lỗi đã đọc:** SH015, BM25 top-5/context 4.000 vẫn không đưa đoạn mô tả gió/mưa vào context và agent từ chối sau 3 search; cần query expansion theo từ đồng nghĩa. SH009, BM25 nhiều lượt bị kéo sang nhánh “murder” và chỉ trả lời một phần câu hỏi về việc phá tượng Napoleon; grade 1. SH013 trả lời “South African securities” thay vì chi tiết “gold fields”; grade 1. Tăng top-k/context nâng Context Evidence dev của BM25 từ 10/15 lên 12/15, nhưng các lỗi paraphrase và lựa chọn bằng chứng vẫn còn.
- **Giới hạn phần cơ bản:** corpus văn học tiếng Anh; test cơ bản có 5 câu có đáp án; câu thiếu evidence dựa trên kiểm tra cụm từ; dense dev dùng trace Day 6 cũ; chưa chạy dense trên test mới hoặc agent dense. Venv riêng đã tạo nhưng dừng cài dependencies vì tốc độ tải thấp; lệnh tiếp tục cài nằm trong README.

## Tái lập

```powershell
python -m unittest discover -s tests -v
python submission/build_dataset.py
python submission/run_lexical.py --split dev --output submission/results/lexical-dev
python submission/run_lexical.py --split dev --top-k 5 --budget-chars 4000 --output submission/results/bm25-top5-dev
python submission/run_hybrid_cached.py --bm25-traces submission/results/lexical-dev/traces.json --output submission/results/hybrid-dev
python submission/run_lexical.py --split test --top-k 5 --budget-chars 4000 --output submission/results/lexical-top5-test
```

Trace và summary ở `submission/results/`; bộ câu hỏi, chunk và văn bản nguồn ở `data/`.

## Thử thách nâng cao — báo cáo World Bank

- **Tài liệu:** World Bank, *World Development Report 2021: Data for Better Lives* (bản cuối 2021, PDF ISBN 978-1-4648-1601-7, DOI 10.1596/978-1-4648-1600-0), 350 trang vật lý. Nguồn chính thức: [PDF World Bank](https://documents1.worldbank.org/curated/en/248201616598597113/pdf/World-Development-Report-2021-Data-for-Better-Lives.pdf). Tài liệu ghi giấy phép CC BY 3.0 IGO; cần xem lưu ý thành phần bên thứ ba trước khi tái sử dụng riêng từng hình.
- **Index:** trích toàn bộ bằng `pdftotext -layout`; index chia theo trang và đoạn, giữ `source_id`, số trang PDF, và liên kết figure/table/map qua văn bản, tiêu đề, chú giải và nhãn trích xuất. Thống kê chi tiết, SHA-256 và các trang lỗi nằm trong `data/advanced/source_manifest.json` và `pages.jsonl`. Có 15 trang không có text máy đọc được, chưa OCR; vùng visual được kiểm chứng bằng trang PDF gốc, các ảnh QA cục bộ nằm trong `data/advanced/evidence_pages/` và không cần để tái tạo.
- **Benchmark:** 18 dev (15 có đáp án, 3 thiếu evidence) và 30 test (24 có đáp án, 6 thiếu evidence). Test gồm bảng 6, ảnh/sơ đồ 5, biểu đồ 5, multi-hop 4, text 4; câu thiếu evidence 6. Câu và evidence chuẩn được tạo trước khi chạy, không đưa vào agent.
- **Cấu hình khóa trước test:** DeepSeek `deepseek-flash`; BM25 có ưu tiên locator Figure/Map/Table; top-k 8; context tối đa 18.000 ký tự; tối đa 3 lần search. Hash cấu hình và input trong `submission/results/advanced_locked_config.json`.
- **Kết quả test:** **29/30 = 96.67%**, đạt ngưỡng 90%; 23/24 câu có đáp án và 6/6 câu thiếu evidence đạt. Theo nhóm: bảng 6/6, ảnh/sơ đồ 5/5, biểu đồ 5/5, multi-hop 3/4, text 4/4, thiếu evidence 6/6. Không có lỗi hệ thống. WDR-T19 là lỗi duy nhất: câu trả lời được tài liệu hỗ trợ nhưng agent từ chối.
- **Thời gian và chi phí:** tổng thời gian agent ghi nhận 72.66 giây; 67 requests, 160,423 tokens; chi phí ước tính **$0.019459 USD** theo giá DeepSeek Flash off-peak tại thời điểm chạy. Đây là ước lượng từ usage, không phải hóa đơn cuối cùng.
- **Đối chiếu:** từng câu test có nhãn đạt/không đạt, lý do, trang evidence chuẩn và trang được trích trong `submission/results/advanced-bm25-test-final/reviews.json`; trace đầy đủ trong `traces.json`, metrics nhóm trong `metrics.json`. Các bảng, biểu đồ, sơ đồ và bản đồ trong benchmark đã được đối chiếu với trang PDF gốc.

Lệnh dựng index, benchmark, chạy dev/test và tính metrics nằm trong mục Advanced của `submission/README.md`. `.env`, bản PDF gốc và ảnh QA không được commit; người chấm có thể tải lại PDF từ nguồn chính thức theo manifest.

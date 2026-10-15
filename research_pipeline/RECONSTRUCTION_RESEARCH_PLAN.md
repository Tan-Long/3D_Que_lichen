# Kế hoạch thử nghiệm tái dựng bề mặt thân cây

Last updated: 2026-10-11.

## Mục tiêu

Thử hai hướng cải thiện mesh RGB-D sau khi batch TSDF hiện tại hoàn tất:

1. TSDF có trọng số theo độ tin cậy.
2. Trải mặt trụ và triangulation có kiểm chứng quan sát.

Mục tiêu là giữ tối đa bề mặt thực sự đã được camera đo. Không tự lấp, suy diễn hoặc tạo bề mặt tại vùng chưa có bằng chứng depth.

Point cloud hoặc surfel tích lũy chỉ là lớp chẩn đoán để xác nhận camera đã đo vùng nghi ngờ hay chưa; đây không phải hướng dựng mesh chính.

## Trạng thái

- [x] Xác định hai hướng nghiên cứu chính.
- [x] Xác định lớp chẩn đoán point cloud hoặc surfel.
- [ ] Chờ batch TSDF hiện tại hoàn tất và giữ nguyên output làm baseline.
- [ ] Chọn năm mẫu thử: ba mẫu bị hở và hai mẫu QA tốt. Đã chạy smoke test trên một mẫu hở và một mẫu QA tốt.
- [x] Triển khai phép đo support quan sát và reprojection error cho prototype.
- [x] Spike khả năng tích hợp TSDF với trọng số theo pixel.
- [x] A/B tăng cường TSDF theo vùng observed support yếu trên ba mẫu.
- [ ] Chạy ma trận thử nghiệm TSDF có trọng số.
- [ ] Prototype trải trụ và Delaunay có visibility mask.
- [ ] So sánh A/B và quyết định phương án triển khai hàng loạt.

## Smoke test ngày 2026-10-12

Đã chạy tuần tự trên hai mẫu, không ghi đè baseline:

- Mẫu hở `cay_0002_2105_CL-2__StrayScanner_export_20260522_231359`.
- Mẫu control đã được QA thủ công duyệt `26Q13`.

### Hướng 1: TSDF theo confidence

Open3D 0.20.0 không nhận weight map theo pixel trong
`VoxelBlockGrid.integrate()`. Không dùng cách lặp frame để giả trọng số. Muốn thử đúng
weighted TSDF phải viết custom integrator hoặc đổi sang backend có weight map thật.

Ablation hợp lệ với ngưỡng cứng `confidence=2` trên mẫu hở cho kết quả:

| Biến thể | Tam giác | Diện tích m² | Hit fraction | P90 residual mm |
| --- | ---: | ---: | ---: | ---: |
| Baseline confidence 1 | 7.989 | 0,0736 | 0,9884 | 4,23 |
| Confidence 2 | 6.251 | 0,0570 | 0,9672 | 4,01 |

Lọc cứng làm mất thêm bề mặt dù residual giảm nhẹ. Kết quả này không thay thế thử
nghiệm trọng số mềm, nhưng loại `confidence=2` khỏi danh sách phương án sửa gap.

### Hướng 2: lớp quan sát đối chứng

Prototype `tools/build_observed_cylinder_pilot.py` xuất `observed_points.ply` và đếm
support theo frame độc lập, không đếm nhiều pixel trong cùng frame như nhiều quan sát.

| Mẫu | Frame kiểm tra | Voxel quan sát | Voxel ≥2 frame | Support trung vị |
| --- | ---: | ---: | ---: | ---: |
| CL-2 | 52 | 10.863 | 7.784 | 4 frame |
| 26Q13 | 177 | 88.651 | 53.495 | 2 frame |

Lớp này dùng được để phân biệt vùng camera không đo với vùng TSDF bỏ sót. Nó vẫn chỉ
là dữ liệu chẩn đoán, không phải bề mặt để đo diện tích.

### Hướng 3: trải trụ và triangulation có depth support

Prototype chỉ giữ điểm được ít nhất hai frame quan sát, giới hạn cạnh/góc/chiều cao
tam giác và chiếu tâm mỗi tam giác về depth; tam giác phải được ít nhất hai frame xác
nhận trong sai số 12 mm.

| Mẫu | Tam giác giữ/ứng viên | Hit baseline → thử | Diện tích baseline → thử |
| --- | ---: | ---: | ---: |
| CL-2 | 15.439 / 15.453 | 0,9884 → 0,9974 | 0,0736 → 0,1963 m² |
| 26Q13 | 101.950 / 102.013 | 0,8680 → 0,9632 | 0,4249 → 2,0579 m² |

**Kết luận tạm thời:** chưa đạt. Depth support làm tăng độ phủ chiếu lại nhưng không
ngăn được bề mặt gợn và diện tích phồng 2,7–4,8 lần. Không dùng output này để đo diện
tích hoặc thay baseline. Muốn tiếp tục phải thêm ràng buộc normal/radius consistency
và đánh giá diện tích ổn định trước khi mở rộng đủ năm mẫu.

Output smoke test nằm trong `artifacts/reconstruction_research_pilot/`.

## A/B voxel 4 mm và frame step 2 ngày 2026-10-12

Đã chạy ứng viên `tensor_v4mm_step2_conf1_adaptive_r35` trên ba mẫu deferred có
hit-rate thấp và hai control QA tốt. Không mẫu deferred nào chuyển thành đạt. Hai
control vẫn đạt, nhưng số tam giác tăng khoảng 59–61% và diện tích tăng 2,2–2,9%.
Hit-rate thay đổi từ -2,18 đến +0,35 điểm phần trăm trên ba mẫu lỗi.

**Quyết định:** không thay baseline và không mở rộng ứng viên này lên toàn bộ dataset.
Chỉ giảm voxel và tăng số frame không giải quyết được nguyên nhân chính. Báo cáo đầy
đủ nằm tại `artifacts/reconstruction_pipeline_pilot/REPORT.md`.

## A/B tăng cường theo observed support ngày 2026-10-12

Pipeline giữ pass TSDF step 3 làm nền. Các frame xen kẽ chỉ được tích hợp tại pixel
depth nằm trong 7,5 mm quanh voxel observed có support dưới hai frame. Mỗi scan giữ
12 frame độc lập cho heldout QA. Vùng không có depth thật không được nội suy hoặc vá.

| Mẫu | Weak point gần mesh ≤7,5 mm | Weak p90 mm | Boundary edge | Azimuth p10 | Hit fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| CL-2 | 56,6% → 61,3% | 13,78 → 12,53 | 507 → 544 | 18,3% → 20,6% | 98,837% → 98,872% |
| 26Q52 | 61,6% → 64,4% | 17,05 → 16,38 | 4.974 → 4.471 | 88,9% → 94,4% | 95,960% → 96,152% |
| 26Q13 control | 32,0% → 33,9% | 22,89 → 22,47 | 772 → 702 | 100% → 100% | 86,796% → 87,224% |

**Kết luận tạm thời:** cả ba mẫu đều đưa mesh gần vùng quan sát yếu hơn và không làm
heldout residual xấu đi đáng kể. Hai mẫu giảm boundary edge; CL-2 tăng 507 lên 544,
nên chưa chạy hàng loạt trước khi QA trực quan thêm. Runner đã hỗ trợ
`--observed-support-root` và ghi sang run name `observed_boost_v1`, không đè baseline.

### Kết luận vision sau batch

Đã xem lại bằng ảnh render toàn bộ 140 mẫu người dùng đánh dấu `Duyệt` và 145 cặp
baseline/boost. Bốn mẫu chuyển deferred → pass theo metric đều không đủ điều kiện
promote bằng hình ảnh: lỗ/lớp kép vẫn còn hoặc boost thêm mảng nền. Do đó observation
chỉ giữ vai trò chẩn đoán/ranking support; chưa dùng để tự động thay mesh production.
Báo cáo và danh sách cần xem lại nằm tại
`artifacts/vision_qa/VISION_QA_DEEP_DIVE.md`.

## Baseline phải giữ nguyên

Không ghi đè output hiện tại. Cấu hình baseline:

- Open3D Tensor `VoxelBlockGrid`.
- Voxel size: 5 mm.
- SDF truncation: 20 mm.
- Frame step: 3.
- Confidence tối thiểu: 1.
- Depth tối đa: 2 m.
- ROI thân cây thích nghi, bán kính tìm kiếm: 0,35 m.
- Hai worker là giới hạn đã thử trên máy Mac 16 GB RAM.

Với mỗi mẫu A/B, giữ nguyên input, pose, intrinsics, ROI và hậu xử lý để chỉ đo tác động của thuật toán tái dựng.

## Lớp chẩn đoán quan sát

Xuất point cloud hoặc surfel tích lũy từ toàn bộ frame depth hợp lệ. Thành phần này không nối mặt và không được dùng làm output mesh chính.

Diễn giải kết quả:

- Không thấy vùng hở trong dữ liệu tích lũy: camera không đo được, vùng bị mask loại hoặc camera chưa quay tới.
- Có điểm nhất quán nhưng TSDF không có mesh: kiểm tra tích hợp TSDF, ngưỡng weight và bước extract mesh.
- Có nhiều lớp điểm lệch nhau: kiểm tra pose và đồng bộ frame trước khi đổi thuật toán dựng.
- Chỉ có điểm rời rạc hoặc không đồng thuận: coi là nhiễu, không phải bằng chứng đủ để tạo mặt.

Viewer QA cần cho phép bật hoặc tắt mesh và lớp quan sát đối chứng trên cùng hệ tọa độ.

## Tiêu chí đánh giá chung

- **Observed support:** số frame depth độc lập hỗ trợ mỗi đỉnh hoặc tam giác.
- **Observed completeness:** tỷ lệ vùng có đủ quan sát nhưng không sinh mesh.
- **Depth reprojection error:** sai số giữa mesh render và depth đo trên các frame kiểm tra.
- **Geometry stability:** nhiễu bề mặt, mặt đảo, tam giác dài và thành phần rời.
- **Performance:** thời gian mỗi mẫu, RAM đỉnh và kích thước output.
- **No hallucination:** không có tam giác tại vùng thiếu số quan sát tối thiểu.

Không dùng độ kín của mesh làm thước đo chất lượng. Mesh kín có thể là dấu hiệu đã tạo mặt tại vùng chưa quan sát.

## Hướng 1 TSDF có trọng số theo độ tin cậy

### Mục tiêu

Thay quyết định giữ hoặc bỏ cứng bằng trọng số mềm. Quan sát chất lượng thấp đóng góp ít; nhiều quan sát tốt và đồng thuận đóng góp mạnh. Voxel chưa được quan sát không được sinh mặt.

Trọng số dự kiến:

```text
w = w_confidence * w_depth * w_angle * w_edge * w_consistency
```

- `w_confidence`: ánh xạ confidence cảm biến thành trọng số.
- `w_depth`: giảm trọng số khi khoảng cách và nhiễu chiều sâu tăng.
- `w_angle`: giảm trọng số của quan sát quá xiên.
- `w_edge`: giảm hoặc loại pixel sát discontinuity depth.
- `w_consistency`: giảm điểm không đồng thuận với các frame lân cận sau khi áp dụng pose.

### Pilot lọc đồng thuận theo frame (2026-10-12)

Đã triển khai spike `--frame-consistency-filter` và thử trên ba mẫu vỏ kép. Cách
lọc cả frame chỉ cải thiện một phần `26Q90`; hai mẫu còn lại mất coverage hoặc
residual xấu hơn. Không đưa vào production. Kết quả đầy đủ nằm tại
`artifacts/frame_consistency_pilot/REPORT.md`. Hướng tiếp theo là lọc pixel depth
cục bộ theo đồng thuận hoặc refine pose trước fusion, không tiếp tục hạ ngưỡng lọc
cả frame.

### Spike kỹ thuật

- [ ] Kiểm tra phiên bản Open3D đang cài có nhận trọng số tùy ý theo pixel trong `VoxelBlockGrid.integrate()` hay không.
- [ ] Nếu API chỉ cộng trọng số cố định, không giả lập trọng số bằng cách lặp frame.
- [ ] Xác định thay đổi nhỏ nhất: cập nhật trực tiếp TSDF và weight trong block, hoặc thử backend cho phép weighted integration.
- [ ] Không đổi backend trước khi spike chứng minh là cần thiết.

### Thứ tự thử

1. Baseline hiện tại.
2. Giữ frame step 3; thêm `w_confidence` và `w_depth`.
3. Thêm `w_edge`.
4. Nếu tốt hơn baseline, thử frame step 2 rồi step 1.
5. Sau cùng mới thêm `w_angle` và `w_consistency`, vì chúng phụ thuộc pháp tuyến và pose.

### Điều kiện đạt

- Giảm vùng thiếu mesh tại nơi lớp chẩn đoán chứng minh đã được quan sát.
- Sai số chiếu lại không tăng đáng kể so với phân bố baseline.
- Không xuất tam giác thiếu observed support tối thiểu.
- Không làm mất rõ rệt chi tiết vỏ cây so với baseline 5 mm.
- RAM và thời gian vẫn cho phép chạy hai worker trên máy hiện tại.

## Hướng 2 Trải mặt trụ và triangulation

### Phạm vi

Chỉ áp dụng cho thân cây hoặc từng đoạn thân gần hình trụ. Không áp dụng trực tiếp cho chạc cây, cành phức tạp hoặc đoạn lệch quá xa mô hình trụ.

### Quy trình dự kiến

1. Tạo point cloud hợp nhất đã lọc nhiễu nhưng chưa dựng mesh.
2. Ước lượng đường tâm theo các lát chiều cao và tạo hệ trục cục bộ cho thân cong.
3. Chuyển điểm sang tọa độ trụ cục bộ `(theta, z, r)`.
4. Xử lý seam `theta = -pi/pi` bằng các điểm biên nhân bản chỉ trong bước triangulation.
5. Chạy Delaunay trong mặt phẳng `(theta, z)` nhưng giữ tọa độ 3D đo được làm đỉnh.
6. Loại tam giác có cạnh 3D quá dài, khoảng góc hoặc chiều cao quá lớn hay pháp tuyến bất thường.
7. Chiếu tam giác về các frame depth nguồn và chỉ giữ khi đủ frame đồng thuận.
8. Ghép các đoạn lân cận nhưng không nối qua chạc hoặc khoảng trống quan sát.

### Điều kiện đạt

- Bảo toàn chi tiết và màu tại các điểm đã đo.
- Không nối tắt qua mặt sau chưa quay hoặc qua khoảng trống lớn.
- Không tạo dải tam giác tại seam góc.
- Phát hiện đoạn không phù hợp mô hình trụ và trả đoạn đó về TSDF baseline.
- Chỉ thay TSDF tại các đoạn mà phương pháp này tốt hơn theo cả metrics và QA.

## Ma trận thử nghiệm ban đầu

| Biến thể | Mục đích | Số mẫu |
| --- | --- | ---: |
| Baseline TSDF | Mốc so sánh | 5 |
| TSDF với confidence và depth weight | Kiểm tra lợi ích chính | 5 |
| TSDF thêm lọc biên depth | Giảm rách và mặt ma | 5 |
| TSDF tốt nhất với frame step 1 hoặc 2 | Đo lợi ích của nhiều frame | 3 |
| Trải trụ và Delaunay có visibility mask | Backend riêng cho đoạn thân phù hợp | 5 |

## Quy tắc quyết định

- Chọn TSDF có trọng số làm pipeline chính nếu tăng observed completeness mà không tăng unsupported surface.
- Chỉ dùng trải trụ cho đoạn đạt điều kiện hình học và vượt kiểm tra visibility.
- Có thể dùng phương án lai: TSDF mặc định, trải trụ thay thế có điều kiện cho phần thân phù hợp.
- Không chạy lại toàn bộ dataset trước khi hoàn tất A/B và QA trên năm mẫu.

## Sản phẩm cần có

- Một lệnh thử nghiệm không ghi đè output hiện tại.
- Mesh của từng biến thể và point cloud hoặc surfel đối chứng.
- Metrics theo mẫu: coverage, observed support, reprojection error, RAM và thời gian.
- Viewer QA hiển thị mesh, dữ liệu quan sát và tam giác thiếu hỗ trợ bằng các lớp riêng.
- Báo cáo ngắn ghi phương án thắng, trường hợp thất bại và cấu hình cuối.

## Tài liệu tham khảo

- [Improving 3D Reconstruction Through RGB-D Sensor Noise Modeling](https://www.mdpi.com/1424-8220/25/3/950)
- [Robust and Efficient CPU-Based RGB-D Scene Reconstruction](https://pmc.ncbi.nlm.nih.gov/articles/PMC6263609/)
- [A Volumetric Method for Building Complex Models from Range Images](https://aperture.stanford.edu/papers/volrange/paper_1_level/paper.html)
- [Tree trunk reconstruction by cylindrical projection and Delaunay triangulation](https://caf.irtree.com/articles/article_detail.aspx?id=0043b6b66caa46ebae9d85a0e4b94117)

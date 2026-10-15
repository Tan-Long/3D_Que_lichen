# Định lượng sinh vật bám trên thân quế

Ngôn ngữ chuẩn dùng khi thiết kế và đánh giá pipeline dựng bề mặt, nhận diện sinh vật bám và định lượng mức che phủ trên thân cây quế.

## Language

**Vùng đo hợp lệ**:
Phần bề mặt thân cây được quan sát đủ chất lượng và được đưa vào mẫu số khi tính tỷ lệ che phủ.
_Avoid_: Toàn bộ cây, toàn bộ mesh

**Diện tích bề mặt**:
Diện tích ba chiều của vùng đo hợp lệ trên thân cây, biểu diễn bằng cm².
_Avoid_: Diện tích ảnh, diện tích chiếu

**Diện tích địa y**:
Phần diện tích bề mặt trong vùng đo hợp lệ được xác định là địa y theo bộ nhãn tham chiếu.
_Avoid_: Số pixel địa y

**Tỷ lệ che phủ địa y**:
Tỷ số giữa diện tích địa y và diện tích bề mặt của cùng một vùng đo hợp lệ, biểu diễn theo phần trăm.
_Avoid_: Độ phủ ảnh

**Sinh vật bám gây nhiễu**:
Rêu, tảo và nấm khác có thể giống địa y về màu sắc hoặc cấu trúc nhưng không thuộc lớp địa y mục tiêu.
_Avoid_: Nhiễu màu

**Hệ nhãn sinh vật bám**:
Hệ nhãn phân cấp gồm địa y, rêu, tảo, nấm khác, vỏ cây và không chắc chắn; địa y và vỏ cây là hai lớp bắt buộc.
_Avoid_: Nhãn màu

**Nhãn tham chiếu**:
Nhãn thủ công dùng làm chuẩn đánh giá, được tạo theo quy trình nhất quán và tách biệt với dữ liệu huấn luyện.
_Avoid_: Ground truth tuyệt đối

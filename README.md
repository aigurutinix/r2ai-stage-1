<p align="center">
  <img src="./images/aiguru_icon.png" alt="AI Guru Logo" height="90" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./images/r2ai_icon.png" alt="R2AI Logo" height="90" />
</p>

<p align="center">
  <b><a href="#tiếng-việt">Tiếng Việt</a></b> | <b><a href="#english">English</a></b>
</p>

---

<a name="tiếng-việt"></a>
# Road to AI (R2AI) – Stage 1

## 1. Giới thiệu cuộc thi

**Road to AI (R2AI)** là cuộc thi và cộng đồng về AI Engineering uy tín tại Việt Nam do **AI Guru** tổ chức. Cuộc thi được xây dựng với mục tiêu thúc đẩy nghiên cứu và phát triển trong lĩnh vực xử lý ngôn ngữ tự nhiên tiếng Việt, đặc biệt tập trung vào các bài toán thực tiễn trong doanh nghiệp. **R2AI – Stage 1** mang đến thử thách xây dựng hệ thống **Legal AI Assistant** hỗ trợ tra cứu và hỏi đáp pháp luật, giúp các kỹ sư AI và nhà phát triển tối ưu hóa quy trình nghiệp vụ pháp lý thông qua công nghệ trí tuệ nhân tạo.

### Bối cảnh bài toán
Doanh nghiệp SME tại Việt Nam thường gặp khó khăn trong việc tra cứu và áp dụng các quy định pháp lý liên quan đến Luật Doanh nghiệp, thuế, lao động, hợp đồng... Trợ lý pháp lý AI cho doanh nghiệp được xây dựng nhằm hỗ trợ chủ doanh nghiệp, kế toán, nhân sự tra cứu nhanh các điều luật, hỏi đáp tình huống pháp lý cụ thể và nhận tư vấn sơ bộ dựa trên hệ thống văn bản pháp luật chính thống.

Trong bối cảnh trí tuệ nhân tạo phát triển mạnh mẽ, đặc biệt với sự xuất hiện của các mô hình ngôn ngữ lớn như ChatGPT, DeepSeek và Qwen, nhu cầu xây dựng các hệ thống AI hỗ trợ xử lý văn bản pháp luật ngày càng trở nên quan trọng. Tuy nhiên, so với các ngôn ngữ như tiếng Anh, tiếng Nhật hay tiếng Trung, nguồn tài nguyên và các nghiên cứu về Vietnamese Legal NLP vẫn còn hạn chế.

Nhằm thúc đẩy nghiên cứu và phát triển trong lĩnh vực này, chúng tôi tổ chức cuộc thi về **Truy hồi và Hỏi đáp Văn bản Pháp luật Tiếng Việt (Vietnamese Legal Information Retrieval & Question Answering)**. Cuộc thi hướng tới việc xây dựng các hệ thống AI có khả năng tìm kiếm điều luật liên quan và tự động trả lời các câu hỏi pháp lý dựa trên căn cứ pháp luật.

### Truy hồi thông tin (Information Retrieval - IR)
Truy hồi thông tin (Information Retrieval - IR) là một nhiệm vụ cốt lõi trong NLP, liên quan đến việc xác định thông tin nào phù hợp nhất với một truy vấn cho trước. Trong lĩnh vực pháp luật, nhiệm vụ Truy hồi Văn bản Pháp luật tập trung vào việc xác định điều luật nào có liên quan đến một câu hỏi pháp lý cụ thể. 

Nhiệm vụ có thể được hình thức hóa như sau: Cho một tập câu hỏi $Q = \{q_1, q_2, ..., q_n\}$ và một kho điều luật $A = \{a_1, a_2, ..., a_m\}$, nhiệm vụ yêu cầu xác định một tập con $A' \subset A$ trong đó mỗi điều luật $a_i \in A'$ được coi là "liên quan" đến câu hỏi tương ứng $q$. Chúng tôi gọi một điều luật là "Liên quan" đến một truy vấn nếu câu truy vấn có thể được trả lời Có/Không, được suy ra từ ý nghĩa của điều luật đó.

### Hỏi đáp pháp luật (Legal Question Answering - QA)
Dựa trên các điều luật đã được truy hồi, hệ thống cần sinh ra câu trả lời cho câu hỏi pháp lý tương ứng. Mục tiêu của nhiệm vụ là xây dựng các hệ thống AI có khả năng không chỉ tìm đúng căn cứ pháp luật mà còn hiểu và suy luận nội dung pháp lý để hỗ trợ trả lời tự động cho người dùng.

### Mục tiêu cuộc thi
Các đội thi cần xây dựng hệ thống AI có khả năng:
1. **Tra cứu pháp lý chính xác**:
   * Tra cứu điều khoản trong Luật Doanh nghiệp và các văn bản liên quan đến SME.
   * Tìm kiếm và truy xuất thông tin pháp luật chính xác từ kho dữ liệu được cung cấp.
   * Ưu tiên khả năng retrieval và grounding chính xác.
2. **Hỏi đáp pháp lý bằng tiếng Việt**:
   * Hiểu ngôn ngữ tự nhiên tiếng Việt.
   * Hỏi đáp các tình huống pháp lý thường gặp.
3. **Dẫn nguồn điều luật**:
   * Trích dẫn điều/khoản/văn bản liên quan.
   * Hiển thị rõ nguồn tham chiếu để đảm bảo khả năng kiểm chứng thông tin.
   * Hạn chế việc trả lời không có căn cứ pháp lý.
4. **Tư vấn sơ bộ & cảnh báo giới hạn**:
   * Đưa ra hướng dẫn pháp lý sơ bộ cho người dùng.
   * Nhắc nhở các rủi ro tuân thủ trong các tình huống phổ biến.
   * Hiển thị cảnh báo giới hạn AI.
5. **Kiểm soát nội dung sai lệch**:
   * Hạn chế việc AI sinh ra thông tin sai lệch.
   * Tránh bịa điều luật hoặc nguồn tham chiếu không tồn tại.
   * Tăng độ tin cậy của câu trả lời dựa trên dữ liệu được cung cấp.

---

## 2. Kết quả cuộc thi & Danh sách các đội

Dưới đây là bảng vinh danh kết quả các đội thi xuất sắc nhất tại R2AI Stage 1 cùng liên kết đến mã nguồn và dữ liệu tương ứng của từng đội:

| Hạng giải | Đội thi | Thư mục dự án |
| :--- | :--- | :--- |
| 🥇 **Giải Nhất** | mscAI | [mscai](./mscai) |
| 🥈 **Giải Nhì** | Hung&Fong | [hung&phong](./hung&phong) |
| 🥉 **Giải Ba** | Nguyễn Văn Nghiêm | [nguyenvannghiem](./nguyenvannghiem) |
| 🏅 **Giải Khuyến khích** | TQD | [tqd](./tqd) |
| 🏅 **Giải Khuyến khích** | FAI Team | [faiteam](./faiteam) |
| 🏅 **Giải Khuyến khích** | Agentic Builders | [agentic_builders](./agentic_builders) |
| 🏅 **Giải Khuyến khích** | NextGen | [nextgen](./nextgen) |
| 🏅 **Giải Khuyến khích** | BeeIT | [beeit](./beeit) |
| 🏅 **Giải Khuyến khích** | Trần Thanh Tú | [tranthanhtu](./tranthanhtu) |
| 🏅 **Giải Khuyến khích** | Thanh Khâu Sơn | [thanhkhauson](./thanhkhauson) |

*Mỗi thư mục dự án của các đội đã được chuẩn hóa theo cấu trúc thống nhất gồm thư mục `src` chứa mã nguồn và `data` chứa dữ liệu, kèm theo tệp hướng dẫn riêng.*

---

## 3. Thông tin liên hệ Ban Tổ chức
**AI Guru – Công ty Cổ phần Tập đoàn Dagoras Group**
* **Địa chỉ**: Tầng 8, số 80 Duy Tân, Cầu Giấy, Hà Nội
* **Đầu mối liên hệ**:
  * **Nguyễn Thị Minh Nguyệt**: Điện thoại: `0981544974` | Email: `nguyetntm@dagoras.io`
  * **Vũ Thị Thuỳ Linh**: Điện thoại: `0961891198` | Email: `linhvtt@dagoras.io`
* **Fanpage**: [AI Guru](https://www.facebook.com/AIGuru.vn)
* **Website**: [r2ai.aiguru.com.vn](https://r2ai.aiguru.com.vn)

<p align="right">
  <a href="#tiếng-việt">Lên đầu trang</a>
</p>

---

<a name="english"></a>
# Road to AI (R2AI) – Stage 1

## 1. Introduction

**Road to AI (R2AI)** is a prestigious AI Engineering competition and community in Vietnam organized by **AI Guru**. The competition is established with the goal of promoting research and development in Vietnamese Natural Language Processing (NLP), with a particular focus on practical business applications. **R2AI – Stage 1** presents the challenge of building a **Legal AI Assistant** for legal document retrieval and question answering, helping AI engineers and developers optimize legal workflows through artificial intelligence technology.

### Problem Context
SMEs (Small and Medium Enterprises) in Vietnam often face difficulties in searching for and applying legal regulations related to Corporate Law, taxes, labor, contracts, etc. The AI Legal Assistant for businesses is built to support business owners, accountants, and HR personnel in quickly searching for laws, answering specific legal scenarios, and receiving preliminary advice based on official legal documents.

In the context of the rapid development of artificial intelligence, especially with the emergence of Large Language Models (LLMs) such as ChatGPT, DeepSeek, and Qwen, the need to build AI systems to assist in processing legal documents is becoming increasingly important. However, compared to languages like English, Japanese, or Chinese, resources and research on Vietnamese Legal NLP remain limited.

To promote research and development in this field, we organize the competition on **Vietnamese Legal Information Retrieval & Question Answering**. The competition aims to build AI systems capable of searching for relevant laws and automatically answering legal questions based on legal grounds.

### Information Retrieval (IR)
Information Retrieval (IR) is a core task in NLP, involving the identification of which information is most relevant to a given query. In the legal domain, the Legal Document Retrieval task focuses on identifying which legal articles are relevant to a specific legal question. 

The task can be formulated as follows: Given a set of questions $Q = \{q_1, q_2, ..., q_n\}$ and a corpus of legal articles $A = \{a_1, a_2, ..., a_m\}$, the task requires identifying a subset $A' \subset A$ where each legal article $a_i \in A'$ is considered "relevant" to the corresponding question $q$. We define a legal article as "Relevant" to a query if the query can be answered with Yes/No, inferred from the meaning of that article.

### Legal Question Answering (QA)
Based on the retrieved legal articles, the system needs to generate answers to the corresponding legal questions. The goal of the task is to build AI systems capable of not only finding the correct legal grounds but also understanding and reasoning over legal content to support automated answering for users.

### Competition Goals
Competing teams need to build AI systems capable of:
1. **Accurate Legal Search**:
   * Search for articles in Corporate Law and documents related to SMEs.
   * Search and retrieve accurate legal information from the provided data corpus.
   * Prioritize retrieval and grounding accuracy.
2. **Vietnamese Legal QA**:
   * Understand natural Vietnamese language.
   * Answer common legal scenarios.
3. **Legal Citation**:
   * Cite relevant articles/clauses/documents.
   * Clearly display reference sources to ensure information verifiability.
   * Minimize answers without legal basis.
4. **Preliminary Advice & Disclaimers**:
   * Provide preliminary legal guidance to users.
   * Warn of compliance risks in common scenarios.
   * Display AI disclaimer.
5. **Control Misleading Content**:
   * Minimize the generation of false information by AI.
   * Avoid hallucinating legal articles or non-existent reference sources.
   * Increase answer reliability based on the provided data.

---

## 2. Competition Results & Team List

Below is the list of the top-performing teams at R2AI Stage 1, along with links to their respective codebases and datasets:

| Award | Team | Project Directory |
| :--- | :--- | :--- |
| 🥇 **First Prize** | mscAI | [mscai](./mscai) |
| 🥈 **Second Prize** | Hung&Fong | [hung&phong](./hung&phong) |
| 🥉 **Third Prize** | Nguyễn Văn Nghiêm | [nguyenvannghiem](./nguyenvannghiem) |
| 🏅 **Consolation Prize** | TQD | [tqd](./tqd) |
| 🏅 **Consolation Prize** | FAI Team | [faiteam](./faiteam) |
| 🏅 **Consolation Prize** | Agentic Builders | [agentic_builders](./agentic_builders) |
| 🏅 **Consolation Prize** | NextGen | [nextgen](./nextgen) |
| 🏅 **Consolation Prize** | BeeIT | [beeit](./beeit) |
| 🏅 **Consolation Prize** | Trần Thanh Tú | [tranthanhtu](./tranthanhtu) |
| 🏅 **Consolation Prize** | Thanh Khâu Sơn | [thanhkhauson](./thanhkhauson) |

*Each team's folder has been standardized into a unified structure containing a `src` directory for source code and a `data` directory for data files, along with its own guide file.*

---

## 3. Organizing Committee Contacts
**AI Guru – Dagoras Group Joint Stock Company**
* **Address**: 8th Floor, No. 80 Duy Tan, Cau Giay, Hanoi
* **Contacts**:
  * **Nguyen Thi Minh Nguyet**: Phone: `0981544974` | Email: `nguyetntm@dagoras.io`
  * **Vu City Thuy Linh**: Phone: `0961891198` | Email: `linhvtt@dagoras.io`
* **Fanpage**: [AI Guru](https://www.facebook.com/AIGuru.vn)
* **Website**: [r2ai.aiguru.com.vn](https://r2ai.aiguru.com.vn)

<p align="right">
  <a href="#english">Back to Top</a>
</p>

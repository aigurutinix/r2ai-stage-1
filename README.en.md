<p align="center">
  <img src="./images/aiguru_icon.png" alt="AI Guru Logo" height="90" />
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="./images/r2ai_icon.png" alt="R2AI Logo" height="90" />
</p>

<p align="center">
  🇻🇳 <a href="./README.md">Tiếng Việt</a> | 🇺🇸 <b>English</b>
</p>

---

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

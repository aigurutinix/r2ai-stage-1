# TÃ€I LIá»†U MÃ” Táº¢ Dá»® LIá»†U

**Dá»± Ã¡n:** Chatbot tra cá»©u vÄƒn báº£n phÃ¡p luáº­t Viá»‡t Nam
**PhiÃªn báº£n tÃ i liá»‡u:** 1.0
**NgÃ y cáº­p nháº­t:** 30/06/2026
**ÄÆ°á»ng dáº«n chia sáº» dá»¯ liá»‡u:** https://drive.google.com/drive/folders/1NvkAEpOjOqHNqF9kC-FvrOs5fjIiscTj

## 1. Má»¥c Ä‘Ã­ch tÃ i liá»‡u

TÃ i liá»‡u nÃ y mÃ´ táº£ bá»™ dá»¯ liá»‡u Ä‘Æ°á»£c sá»­ dá»¥ng Ä‘á»ƒ xÃ¢y dá»±ng vÃ  váº­n hÃ nh há»‡ thá»‘ng chatbot tra cá»©u vÄƒn báº£n phÃ¡p luáº­t Viá»‡t Nam. Ná»™i dung tÃ i liá»‡u phá»¥c vá»¥ cÃ¡c má»¥c tiÃªu:

- Cung cáº¥p thÃ´ng tin vá» nguá»“n gá»‘c vÃ  pháº¡m vi dá»¯ liá»‡u.
- MÃ´ táº£ cáº¥u trÃºc, Ä‘á»‹nh dáº¡ng vÃ  Ã½ nghÄ©a cÃ¡c trÆ°á»ng dá»¯ liá»‡u.
- HÆ°á»›ng dáº«n cÃ¡ch truy cáº­p, sá»­ dá»¥ng vÃ  tÃ¡i láº­p bá»™ dá»¯ liá»‡u.
- Ghi nháº­n cÃ¡c giá»›i háº¡n cháº¥t lÆ°á»£ng dá»¯ liá»‡u vÃ  cÃ¡c lÆ°u Ã½ khi sá»­ dá»¥ng.

TÃ i liá»‡u nÃ y Ä‘Æ°á»£c thiáº¿t káº¿ Ä‘á»ƒ bÃ n giao cho bÃªn thá»© ba cÃ³ thá»ƒ kiá»ƒm tra, sá»­ dá»¥ng láº¡i hoáº·c tÃ¡i láº­p pipeline dá»¯ liá»‡u.

## 2. Tá»•ng quan bá»™ dá»¯ liá»‡u

Bá»™ dá»¯ liá»‡u lÃ  táº­p vÄƒn báº£n phÃ¡p luáº­t Viá»‡t Nam Ä‘Ã£ Ä‘Æ°á»£c xá»­ lÃ½ phá»¥c vá»¥ bÃ i toÃ¡n RAG. ÄÆ¡n vá»‹ truy há»“i chÃ­nh lÃ  **Ä‘iá»u luáº­t/chunk**, khÃ´ng pháº£i toÃ n bá»™ vÄƒn báº£n. Má»—i chunk Ä‘Æ°á»£c gáº¯n metadata vá» vÄƒn báº£n nguá»“n, sá»‘ Ä‘iá»u, tiÃªu Ä‘á», nguá»“n gá»‘c vÃ  ná»™i dung Ä‘á»ƒ phá»¥c vá»¥ truy váº¥n, rerank vÃ  sinh cÃ¢u tráº£ lá»i cÃ³ cÄƒn cá»©.

PhiÃªn báº£n dá»¯ liá»‡u Ä‘ang Ä‘Æ°á»£c sá»­ dá»¥ng trong pipeline má»›i nháº¥t:

| ThÃ nh pháº§n | GiÃ¡ trá»‹ |
|---|---|
| Corpus chÃ­nh | `data/corpus_vbpl_v2_parsefix_20260628/` |
| Qdrant collection | `vbpl_aiteam_meta_parsefix_20260628` |
| BM25 index | `data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl` |
| Embedding model | `AITeamVN/Vietnamese_Embedding_v2` |
| KÃ­ch thÆ°á»›c vector | 1024 |
| Reranker | `AITeamVN/Vietnamese_Reranker` |
| Pipeline sá»­ dá»¥ng | `scripts/build_v72_metadata_arch.py`, `scripts/build_v75_prompt_selector.py`, `scripts/build_v75_v72_selector_repair.py` |

Ghi chÃº: file `.env` hiá»‡n váº«n cÃ²n cáº¥u hÃ¬nh cÅ© `QDRANT_COLLECTION=vbpl_v2`. CÃ¡c pipeline má»›i tá»± Ä‘áº·t collection `vbpl_aiteam_meta_parsefix_20260628` trong code náº¿u biáº¿n mÃ´i trÆ°á»ng chÆ°a Ä‘Æ°á»£c override.

## 3. Nguá»“n dá»¯ liá»‡u

### 3.1. Nguá»“n chÃ­nh

Nguá»“n dá»¯ liá»‡u chÃ­nh lÃ  vÄƒn báº£n phÃ¡p luáº­t tá»«:

- **CÆ¡ sá»Ÿ dá»¯ liá»‡u quá»‘c gia vá» vÄƒn báº£n phÃ¡p luáº­t:** `vbpl.vn`
- **CÆ¡ quan quáº£n lÃ½ nguá»“n:** Bá»™ TÆ° phÃ¡p
- **Dataset trung gian:** HuggingFace `tmquan/vbpl-vn`

Dataset `tmquan/vbpl-vn` lÃ  báº£n scrape cÃ³ cáº¥u trÃºc tá»« `vbpl.vn`, bao gá»“m ná»™i dung vÄƒn báº£n, metadata cÆ¡ báº£n vÃ  Ä‘Æ°á»ng dáº«n vá» vÄƒn báº£n gá»‘c.

CÃ¡c trÆ°á»ng gá»‘c quan trá»ng Ä‘Æ°á»£c sá»­ dá»¥ng:

| TrÆ°á»ng nguá»“n | Ã nghÄ©a |
|---|---|
| `doc_number` | Sá»‘/kÃ½ hiá»‡u vÄƒn báº£n |
| `legal_type` | Loáº¡i vÄƒn báº£n |
| `title` | TÃªn vÄƒn báº£n |
| `issue_date` / `year` | NgÃ y/nÄƒm ban hÃ nh náº¿u cÃ³ |
| `issuing_authority` | CÆ¡ quan ban hÃ nh |
| `legal_area` | LÄ©nh vá»±c phÃ¡p luáº­t |
| `scope` | Pháº¡m vi vÄƒn báº£n |
| `markdown` | Ná»™i dung toÃ n vÄƒn dáº¡ng markdown |
| `source_url` | ÄÆ°á»ng dáº«n vá» vÄƒn báº£n gá»‘c trÃªn `vbpl.vn` |
| `text_hash` | MÃ£ hash ná»™i dung vÄƒn báº£n |

### 3.2. Nguá»“n bá»• sung thá»§ cÃ´ng

NgoÃ i nguá»“n chÃ­nh, há»‡ thá»‘ng sá»­ dá»¥ng thÃªm 11 vÄƒn báº£n `.docx` do nhÃ³m bá»• sung thá»§ cÃ´ng trong thÆ° má»¥c:

```text
data/manual_vbpl/
```

CÃ¡c vÄƒn báº£n nÃ y Ä‘Æ°á»£c bá»• sung khi nguá»“n tá»± Ä‘á»™ng `tmquan/vbpl-vn` thiáº¿u vÄƒn báº£n, thiáº¿u toÃ n vÄƒn, thiáº¿u Ä‘iá»u, hoáº·c parse chÆ°a Ä‘á»§ chÃ­nh xÃ¡c. ToÃ n bá»™ 11 file bá»• sung thá»§ cÃ´ng Ä‘Æ°á»£c nhÃ³m láº¥y tá»« **trang ThÆ° viá»‡n PhÃ¡p luáº­t (`thuvienphapluat.vn`)** dÆ°á»›i dáº¡ng file `.docx`, sau Ä‘Ã³ Ä‘Æ°a vÃ o cÃ¹ng quy trÃ¬nh parse/chunk nhÆ° dá»¯ liá»‡u chÃ­nh.

Danh sÃ¡ch 11 vÄƒn báº£n bá»• sung thá»§ cÃ´ng:

| STT | File trong repo | Sá»‘/kÃ½ hiá»‡u | Loáº¡i vÄƒn báº£n | NÄƒm | CÆ¡ quan ban hÃ nh | TrÃ­ch yáº¿u |
|---:|---|---|---|---:|---|---|
| 1 | `125_2020_ND-CP.docx` | `125/2020/NÄ-CP` | Nghá»‹ Ä‘á»‹nh | 2020 | ChÃ­nh phá»§ | Quy Ä‘á»‹nh xá»­ pháº¡t vi pháº¡m hÃ nh chÃ­nh vá» thuáº¿, hÃ³a Ä‘Æ¡n |
| 2 | `122_2020_ND-CP.docx` | `122/2020/NÄ-CP` | Nghá»‹ Ä‘á»‹nh | 2020 | ChÃ­nh phá»§ | Quy Ä‘á»‹nh vá» phá»‘i há»£p, liÃªn thÃ´ng thá»§ tá»¥c Ä‘Äƒng kÃ½ thÃ nh láº­p doanh nghiá»‡p, chi nhÃ¡nh, vÄƒn phÃ²ng Ä‘áº¡i diá»‡n, khai trÃ¬nh viá»‡c sá»­ dá»¥ng lao Ä‘á»™ng, cáº¥p mÃ£ sá»‘ Ä‘Æ¡n vá»‹ tham gia báº£o hiá»ƒm xÃ£ há»™i, Ä‘Äƒng kÃ½ sá»­ dá»¥ng hÃ³a Ä‘Æ¡n cá»§a doanh nghiá»‡p |
| 3 | `20_2026_TT-BTC.docx` | `20/2026/TT-BTC` | ThÃ´ng tÆ° | 2026 | Bá»™ TÃ i chÃ­nh | HÆ°á»›ng dáº«n Luáº­t Thuáº¿ thu nháº­p doanh nghiá»‡p vÃ  Nghá»‹ Ä‘á»‹nh sá»‘ 320/2025/NÄ-CP |
| 4 | `135_2020_ND-CP.docx` | `135/2020/NÄ-CP` | Nghá»‹ Ä‘á»‹nh | 2020 | ChÃ­nh phá»§ | Quy Ä‘á»‹nh vá» tuá»•i nghá»‰ hÆ°u |
| 5 | `157_2025_ND-CP.docx` | `157/2025/NÄ-CP` | Nghá»‹ Ä‘á»‹nh | 2025 | ChÃ­nh phá»§ | Quy Ä‘á»‹nh chi tiáº¿t vÃ  biá»‡n phÃ¡p thi hÃ nh má»™t sá»‘ Ä‘iá»u cá»§a Luáº­t Báº£o hiá»ƒm xÃ£ há»™i vá» báº£o hiá»ƒm xÃ£ há»™i báº¯t buá»™c Ä‘á»‘i vá»›i quÃ¢n nhÃ¢n, cÃ´ng an nhÃ¢n dÃ¢n vÃ  ngÆ°á»i lÃ m cÃ´ng tÃ¡c cÆ¡ yáº¿u hÆ°á»Ÿng lÆ°Æ¡ng nhÆ° Ä‘á»‘i vá»›i quÃ¢n nhÃ¢n |
| 6 | `81_2018_ND-CP.docx` | `81/2018/NÄ-CP` | Nghá»‹ Ä‘á»‹nh | 2018 | ChÃ­nh phá»§ | Quy Ä‘á»‹nh chi tiáº¿t Luáº­t ThÆ°Æ¡ng máº¡i vá» hoáº¡t Ä‘á»™ng xÃºc tiáº¿n thÆ°Æ¡ng máº¡i |
| 7 | `07_2022.docx` | `07/2022/NÄ-CP` | Nghá»‹ Ä‘á»‹nh | 2022 | ChÃ­nh phá»§ | Sá»­a Ä‘á»•i, bá»• sung má»™t sá»‘ Ä‘iá»u cá»§a cÃ¡c nghá»‹ Ä‘á»‹nh vá» xá»­ pháº¡t vi pháº¡m hÃ nh chÃ­nh trong lÄ©nh vá»±c sá»Ÿ há»¯u cÃ´ng nghiá»‡p; tiÃªu chuáº©n, Ä‘o lÆ°á»ng vÃ  cháº¥t lÆ°á»£ng sáº£n pháº©m, hÃ ng hÃ³a; hoáº¡t Ä‘á»™ng khoa há»c vÃ  cÃ´ng nghá»‡, chuyá»ƒn giao cÃ´ng nghá»‡; nÄƒng lÆ°á»£ng nguyÃªn tá»­ |
| 8 | `07_2022_QH15.docx` | `07/2022/QH15` | Luáº­t | 2022 | Quá»‘c há»™i | Sá»­a Ä‘á»•i, bá»• sung má»™t sá»‘ Ä‘iá»u cá»§a Luáº­t Sá»Ÿ há»¯u trÃ­ tuá»‡ |
| 9 | `68_2026_ND-CP.docx` | `68/2026/NÄ-CP` | Nghá»‹ Ä‘á»‹nh | 2026 | ChÃ­nh phá»§ | Quy Ä‘á»‹nh vá» chÃ­nh sÃ¡ch thuáº¿ vÃ  quáº£n lÃ½ thuáº¿ Ä‘á»‘i vá»›i há»™ kinh doanh, cÃ¡ nhÃ¢n kinh doanh |
| 10 | `141_2026_ND-CP.docx` | `141/2026/NÄ-CP` | Nghá»‹ Ä‘á»‹nh | 2026 | ChÃ­nh phá»§ | Sá»­a Ä‘á»•i, bá»• sung má»™t sá»‘ Ä‘iá»u cá»§a Nghá»‹ Ä‘á»‹nh sá»‘ 68/2026/NÄ-CP quy Ä‘á»‹nh vá» chÃ­nh sÃ¡ch thuáº¿ vÃ  quáº£n lÃ½ thuáº¿ Ä‘á»‘i vá»›i há»™ kinh doanh, cÃ¡ nhÃ¢n kinh doanh |
| 11 | `132_2026_ND-CP.docx` | `132/2026/NÄ-CP` | Nghá»‹ Ä‘á»‹nh | 2026 | ChÃ­nh phá»§ | Sá»­a Ä‘á»•i, bá»• sung má»™t sá»‘ Ä‘iá»u cá»§a Nghá»‹ Ä‘á»‹nh sá»‘ 41/2018/NÄ-CP quy Ä‘á»‹nh xá»­ pháº¡t vi pháº¡m hÃ nh chÃ­nh trong lÄ©nh vá»±c káº¿ toÃ¡n, kiá»ƒm toÃ¡n Ä‘á»™c láº­p |

LÆ°u Ã½: thÆ° má»¥c `data/manual_vbpl/` cÃ²n cÃ³ file phá»¥ trá»£ `125_2020_ND-CP.txt` vÃ  thÆ° má»¥c áº£nh `_png/`; cÃ¡c file nÃ y khÃ´ng Ä‘Æ°á»£c tÃ­nh vÃ o 11 vÄƒn báº£n `.docx` bá»• sung thá»§ cÃ´ng.

Äá»‘i vá»›i nhÃ³m dá»¯ liá»‡u thá»§ cÃ´ng, `source_url` trong payload cÃ³ thá»ƒ Ä‘á»ƒ trá»‘ng do file Ä‘Æ°á»£c nháº­p tá»« tÃ i liá»‡u cá»¥c bá»™. Há»‡ thá»‘ng khÃ´ng tá»± táº¡o link giáº£; provenance Ä‘Æ°á»£c thá»ƒ hiá»‡n qua file nguá»“n, metadata ná»™i bá»™ vÃ  ghi chÃº nguá»“n lÃ  ThÆ° viá»‡n PhÃ¡p luáº­t.

## 4. Pháº¡m vi dá»¯ liá»‡u

Tá»« táº­p dá»¯ liá»‡u gá»‘c, há»‡ thá»‘ng chá»‰ giá»¯ láº¡i cÃ¡c vÄƒn báº£n phÃ¹ há»£p vá»›i pháº¡m vi bÃ i toÃ¡n tra cá»©u phÃ¡p luáº­t doanh nghiá»‡p vÃ  cÃ¡c lÄ©nh vá»±c liÃªn quan.

CÃ¡c tiÃªu chÃ­ lá»c chÃ­nh:

| TiÃªu chÃ­ | MÃ´ táº£ |
|---|---|
| Cáº¥p vÄƒn báº£n | Æ¯u tiÃªn vÄƒn báº£n cáº¥p trung Æ°Æ¡ng, loáº¡i bá» pháº§n lá»›n vÄƒn báº£n Ä‘á»‹a phÆ°Æ¡ng |
| Loáº¡i vÄƒn báº£n | Luáº­t, Bá»™ luáº­t, PhÃ¡p lá»‡nh, Nghá»‹ Ä‘á»‹nh, ThÃ´ng tÆ°, ThÃ´ng tÆ° liÃªn tá»‹ch, VÄƒn báº£n há»£p nháº¥t, Nghá»‹ quyáº¿t, Quyáº¿t Ä‘á»‹nh |
| LÄ©nh vá»±c | Doanh nghiá»‡p, Ä‘áº§u tÆ°, thuáº¿, háº£i quan, lao Ä‘á»™ng, BHXH, thÆ°Æ¡ng máº¡i, cáº¡nh tranh, káº¿ toÃ¡n, chá»©ng khoÃ¡n, báº£o hiá»ƒm, sá»Ÿ há»¯u trÃ­ tuá»‡, phÃ¡ sáº£n, Ä‘áº¥t Ä‘ai, xÃ¢y dá»±ng |
| ÄÆ¡n vá»‹ truy há»“i | TÃ¡ch theo Äiá»u; Ä‘iá»u quÃ¡ dÃ i cÃ³ thá»ƒ Ä‘Æ°á»£c chia nhá» theo khoáº£n/chunk |

Má»¥c tiÃªu cá»§a bÆ°á»›c lá»c lÃ  giáº£m nhiá»…u, tÄƒng Ä‘á»™ chÃ­nh xÃ¡c truy há»“i vÃ  giáº£m chi phÃ­ embedding.

## 5. Cáº¥u trÃºc dá»¯ liá»‡u bÃ n giao

### 5.1. ThÆ° má»¥c corpus chÃ­nh

ThÆ° má»¥c:

```text
data/corpus_vbpl_v2_parsefix_20260628/
```

Ná»™i dung:

| File | MÃ´ táº£ | Quy mÃ´ |
|---|---|---:|
| `documents.parquet` | Metadata cáº¥p vÄƒn báº£n, khÃ´ng chá»©a toÃ n bá»™ ná»™i dung dÃ¹ng cho truy há»“i | 3,926,402 bytes |
| `articles.parquet` | Dá»¯ liá»‡u cáº¥p Ä‘iá»u/chunk, lÃ  táº­p chÃ­nh Ä‘á»ƒ index vÃ o vector database | 146,453,049 bytes |
| `manifest.json` | Danh sÃ¡ch vÄƒn báº£n vÃ  provenance | 21,574 vÄƒn báº£n |
| `inventory.md` | Thá»‘ng kÃª theo loáº¡i vÄƒn báº£n/lÄ©nh vá»±c vÃ  danh sÃ¡ch luáº­t gá»‘c | Markdown |
| `README.md` | MÃ´ táº£ ngáº¯n vá» corpus | Markdown |
| `PARSEFIX_NOTES.md` | Ghi chÃº cÃ¡c sá»­a lá»—i parse Ä‘Ã£ Ã¡p dá»¥ng | Markdown |

### 5.2. Chá»‰ má»¥c vÃ  dá»¯ liá»‡u phá»¥ trá»£

| ÄÆ°á»ng dáº«n | MÃ´ táº£ |
|---|---|
| `data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl` | Chá»‰ má»¥c BM25 dÃ¹ng cho hybrid retrieval |
| `data/manual_vbpl/` | VÄƒn báº£n bá»• sung thá»§ cÃ´ng |
| `data/ft_rerank_train.jsonl` | Dá»¯ liá»‡u huáº¥n luyá»‡n/finetune reranker |
| `docs/QDRANT_DATA_METADATA_2026_06_28.md` | Audit metadata Qdrant |
| `docs/PARSE_CHUNK_FIX_2026_06_28.md` | MÃ´ táº£ chi tiáº¿t sá»­a lá»—i parse/chunk |

## 6. Data dictionary

### 6.1. `manifest.json`

Má»—i pháº§n tá»­ trong `manifest.json` tÆ°Æ¡ng á»©ng vá»›i má»™t vÄƒn báº£n phÃ¡p luáº­t.

| TrÆ°á»ng | Kiá»ƒu dá»¯ liá»‡u | MÃ´ táº£ |
|---|---|---|
| `so_ky_hieu` | string | Sá»‘/kÃ½ hiá»‡u vÄƒn báº£n |
| `loai_van_ban` | string | Loáº¡i vÄƒn báº£n, vÃ­ dá»¥ Luáº­t, Nghá»‹ Ä‘á»‹nh, ThÃ´ng tÆ° |
| `title` | string | TÃªn vÄƒn báº£n |
| `nam` | string/int | NÄƒm ban hÃ nh |
| `linh_vuc` | string | NhÃ³m lÄ©nh vá»±c ná»™i bá»™ sau phÃ¢n loáº¡i |
| `co_quan_ban_hanh` | string | CÆ¡ quan ban hÃ nh |
| `source_url` | string | Link vÄƒn báº£n gá»‘c trÃªn `vbpl.vn`, cÃ³ thá»ƒ trá»‘ng vá»›i dá»¯ liá»‡u thá»§ cÃ´ng |
| `text_hash` | string | Hash ná»™i dung tá»« dataset nguá»“n |
| `char_len` | integer | Äá»™ dÃ i ná»™i dung vÄƒn báº£n gá»‘c |
| `n_dieu` | integer | Sá»‘ Ä‘iá»u parse Ä‘Æ°á»£c |

VÃ­ dá»¥:

```json
{
  "so_ky_hieu": "02/2016/TT-BCT",
  "loai_van_ban": "ThÃ´ng tÆ°",
  "title": "quy Ä‘á»‹nh vá» nguyÃªn táº¯c Ä‘iá»u hÃ nh háº¡n ngáº¡ch thuáº¿ quan nháº­p kháº©u...",
  "nam": "2016",
  "linh_vuc": "Thuáº¿/Háº£i quan",
  "co_quan_ban_hanh": "Bá»™ CÃ´ng ThÆ°Æ¡ng",
  "source_url": "https://vbpl.vn/van-ban/chi-tiet/...",
  "text_hash": "31eac9cfef916a113b68e582798ee158",
  "char_len": 3094,
  "n_dieu": 3
}
```

### 6.2. `documents.parquet`

File nÃ y chá»©a metadata cáº¥p vÄƒn báº£n, phá»¥c vá»¥ kiá»ƒm kÃª, thá»‘ng kÃª vÃ  truy xuáº¥t provenance.

| TrÆ°á»ng | MÃ´ táº£ |
|---|---|
| `so_ky_hieu` | Sá»‘/kÃ½ hiá»‡u vÄƒn báº£n |
| `loai_van_ban` | Loáº¡i vÄƒn báº£n |
| `title` | TÃªn vÄƒn báº£n |
| `nam` | NÄƒm ban hÃ nh |
| `linh_vuc` | LÄ©nh vá»±c ná»™i bá»™ |
| `co_quan_ban_hanh` | CÆ¡ quan ban hÃ nh |
| `source_url` | Link nguá»“n náº¿u cÃ³ |
| `text_hash` | Hash ná»™i dung tá»« nguá»“n |
| `char_len` | Äá»™ dÃ i vÄƒn báº£n |
| `n_dieu` | Sá»‘ Ä‘iá»u parse Ä‘Æ°á»£c |

### 6.3. `articles.parquet`

File nÃ y lÃ  dá»¯ liá»‡u chÃ­nh dÃ¹ng Ä‘á»ƒ táº¡o vector index vÃ  truy há»“i. Má»—i dÃ²ng tÆ°Æ¡ng á»©ng vá»›i má»™t Ä‘iá»u hoáº·c chunk.

| TrÆ°á»ng | MÃ´ táº£ |
|---|---|
| `so_ky_hieu` | Sá»‘/kÃ½ hiá»‡u vÄƒn báº£n chá»©a Ä‘iá»u |
| `loai_van_ban` | Loáº¡i vÄƒn báº£n |
| `title` | TÃªn vÄƒn báº£n |
| `dieu_so` | Sá»‘ Ä‘iá»u |
| `dieu_tieu_de` | TiÃªu Ä‘á» Ä‘iá»u náº¿u parse Ä‘Æ°á»£c |
| `char_len` | Äá»™ dÃ i chunk |
| `text` | Ná»™i dung Ä‘iá»u/chunk |
| `source_url` | Link nguá»“n vÄƒn báº£n náº¿u cÃ³ |

## 7. Dá»¯ liá»‡u trong Qdrant

Collection sá»­ dá»¥ng:

```text
vbpl_aiteam_meta_parsefix_20260628
```

ThÃ´ng tin audit:

| Thuá»™c tÃ­nh | GiÃ¡ trá»‹ |
|---|---|
| Sá»‘ points | 285,002 |
| Vector size | 1024 |
| Metric | Cosine |
| Embedding model | `AITeamVN/Vietnamese_Embedding_v2` |
| BM25 tÆ°Æ¡ng á»©ng | `data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl` |

Payload chÃ­nh trong Qdrant:

| TrÆ°á»ng | MÃ´ táº£ |
|---|---|
| `doc_id` | Äá»‹nh danh ná»™i bá»™ cá»§a vÄƒn báº£n |
| `so_ky_hieu` | Sá»‘/kÃ½ hiá»‡u vÄƒn báº£n |
| `loai_van_ban` | Loáº¡i vÄƒn báº£n |
| `co_quan_ban_hanh` | CÆ¡ quan ban hÃ nh |
| `ngay_ban_hanh` | NgÃ y ban hÃ nh, Ä‘Æ°á»£c enrich tá»« nguá»“n náº¿u cÃ³ |
| `ngay_hieu_luc` | NgÃ y hiá»‡u lá»±c náº¿u cÃ³ |
| `tinh_trang_hieu_luc` | TÃ¬nh tráº¡ng hiá»‡u lá»±c náº¿u cÃ³ |
| `linh_vuc` | LÄ©nh vá»±c |
| `title` | TÃªn vÄƒn báº£n |
| `dieu_so` | Sá»‘ Ä‘iá»u |
| `dieu_tieu_de` | TiÃªu Ä‘á» Ä‘iá»u |
| `khoan_so` | Sá»‘ khoáº£n náº¿u chunk theo khoáº£n |
| `text` | Ná»™i dung chunk |
| `char_len` | Äá»™ dÃ i chunk |
| `source_url` | Link vÄƒn báº£n gá»‘c náº¿u cÃ³ |
| `nguon` | Nguá»“n dá»¯ liá»‡u |
| `nam` | NÄƒm ban hÃ nh |

## 8. Quy trÃ¬nh xá»­ lÃ½ dá»¯ liá»‡u

Quy trÃ¬nh tá»•ng quÃ¡t:

```text
vbpl.vn / tmquan-vbpl-vn
  -> lá»c pháº¡m vi vÄƒn báº£n
  -> parse markdown thÃ nh cáº¥u trÃºc vÄƒn báº£n
  -> tÃ¡ch theo Äiá»u/Khoáº£n
  -> bá»• sung vÄƒn báº£n thá»§ cÃ´ng náº¿u cáº§n
  -> táº¡o manifest vÃ  parquet
  -> embedding báº±ng AITeamVN/Vietnamese_Embedding_v2
  -> lÆ°u vector vÃ  payload vÃ o Qdrant
  -> táº¡o BM25 index
  -> retrieval hybrid + reranker
```

CÃ¡c module chÃ­nh:

| File | Vai trÃ² |
|---|---|
| `ingest/run_vbpl.py` | Ingest dataset vÃ o Qdrant |
| `ingest/parse_vbpl.py` | Parse vÄƒn báº£n phÃ¡p luáº­t |
| `ingest/chunk.py` | TÃ¡ch vÄƒn báº£n thÃ nh chunk |
| `ingest/manual_docs.py` | Náº¡p vÄƒn báº£n thá»§ cÃ´ng |
| `scripts/export_corpus_vbpl.py` | Xuáº¥t corpus ra parquet/manifest |
| `scripts/build_bm25.py` | Táº¡o BM25 index |
| `scripts/build_v72_metadata_arch.py` | Pipeline retrieval cÃ³ metadata |
| `scripts/build_v75_v72_selector_repair.py` | Pipeline má»›i cÃ³ sá»­a selector |

## 9. HÆ°á»›ng dáº«n sá»­ dá»¥ng

### 9.1. Táº£i dá»¯ liá»‡u

Dá»¯ liá»‡u Ä‘Æ°á»£c chia sáº» qua Google Drive:

```text
https://drive.google.com/drive/folders/1NvkAEpOjOqHNqF9kC-FvrOs5fjIiscTj
```

GÃ³i bÃ n giao khuyáº¿n nghá»‹:

```text
data/vbpl_data_package_20260630.zip
```

Sau khi giáº£i nÃ©n, Ä‘áº·t cÃ¡c thÆ° má»¥c/file vá» Ä‘Ãºng cáº¥u trÃºc repo nhÆ° sau:

```text
data/
  corpus_vbpl_v2_parsefix_20260628/
  bm25_vbpl_aiteam_meta_parsefix_20260628.pkl
  manual_vbpl/
  ft_rerank_train.jsonl
docs/
  DATA_DESCRIPTION.md
  QDRANT_DATA_METADATA_2026_06_28.md
  PARSE_CHUNK_FIX_2026_06_28.md
```

### 9.2. Cháº¡y pipeline má»›i nháº¥t

```powershell
$env:QDRANT_COLLECTION="vbpl_aiteam_meta_parsefix_20260628"
$env:QDRANT_URL="http://localhost:6333"
$env:EMBED_BACKEND="st"
$env:EMBED_ST_MODEL="AITeamVN/Vietnamese_Embedding_v2"
$env:HYBRID_SEARCH="true"
$env:USE_RERANKER="true"
$env:RERANKER_MODEL="AITeamVN/Vietnamese_Reranker"
$env:BM25_INDEX_PATH="data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl"

python scripts/build_v75_v72_selector_repair.py
```

### 9.3. Kiá»ƒm tra Qdrant

Qdrant REST API:

```text
http://localhost:6333/collections
http://localhost:6333/collections/vbpl_aiteam_meta_parsefix_20260628
```

## 10. HÆ°á»›ng dáº«n tÃ¡i láº­p dá»¯ liá»‡u

### 10.1. TÃ¡i láº­p corpus parquet

```powershell
$env:CORPUS_OUT="data/corpus_vbpl_v2_parsefix_20260628_export"
python scripts/export_corpus_vbpl.py
```

Lá»‡nh trÃªn Ä‘á»c dataset `tmquan/vbpl-vn`, Ã¡p dá»¥ng bá»™ lá»c pháº¡m vi, parse vÄƒn báº£n vÃ  xuáº¥t cÃ¡c file `documents.parquet`, `articles.parquet`, `manifest.json`, `inventory.md`.

### 10.2. TÃ¡i láº­p Qdrant collection

```powershell
$env:QDRANT_COLLECTION="vbpl_aiteam_meta_parsefix_20260628"
python -m ingest.run_vbpl --recreate
```

Sau khi ingest, cáº§n táº¡o láº¡i BM25 index tÆ°Æ¡ng á»©ng:

```powershell
$env:QDRANT_COLLECTION="vbpl_aiteam_meta_parsefix_20260628"
$env:BM25_INDEX_PATH="data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl"
python scripts/build_bm25.py
```

LÆ°u Ã½: viá»‡c tÃ¡i láº­p Ä‘áº§y Ä‘á»§ phá»¥ thuá»™c vÃ o mÃ´i trÆ°á»ng cÃ³ Qdrant, model embedding, dependency Python vÃ  cache HuggingFace phÃ¹ há»£p.

## 11. Kiá»ƒm soÃ¡t cháº¥t lÆ°á»£ng dá»¯ liá»‡u

CÃ¡c kiá»ƒm tra Ä‘Ã£ thá»±c hiá»‡n vÃ  Ä‘Æ°á»£c ghi nháº­n trong tÃ i liá»‡u audit:

- Kiá»ƒm tra sá»‘ lÆ°á»£ng point trong Qdrant.
- Kiá»ƒm tra cÃ¡c trÆ°á»ng payload báº¯t buá»™c.
- Kiá»ƒm tra sá»‘ lÆ°á»£ng vÄƒn báº£n cÃ³ `source_url`.
- Kiá»ƒm tra enrichment `ngay_ban_hanh`.
- Kiá»ƒm tra integrity giá»¯a collection gá»‘c vÃ  collection metadata.
- Sá»­a lá»—i parse/chunk cÃ³ chá»n lá»c cho má»™t sá»‘ vÄƒn báº£n cÃ³ tÃ¡c Ä‘á»™ng cao.

CÃ¡c sá»­a lá»—i parse trong báº£n `parsefix_20260628`:

| VÄƒn báº£n | Ná»™i dung sá»­a |
|---|---|
| `125/2020/NÄ-CP` | KhÃ´i phá»¥c Äiá»u 29 bá»‹ thiáº¿u do header khÃ´ng Ä‘Ãºng máº«u |
| `41/2024/QH15` | KhÃ´i phá»¥c Äiá»u 38 bá»‹ thiáº¿u do lá»—i OCR/Ä‘á»‹nh dáº¡ng |

## 12. Giá»›i háº¡n vÃ  lÆ°u Ã½ sá»­ dá»¥ng

Bá»™ dá»¯ liá»‡u cÃ³ cÃ¡c giá»›i háº¡n sau:

- Metadata hiá»‡u lá»±c phÃ¡p lÃ½ chÆ°a Ä‘á»§ Ä‘á»ƒ káº¿t luáº­n chÃ­nh xÃ¡c vÄƒn báº£n cÃ²n hiá»‡u lá»±c táº¡i má»™t thá»i Ä‘iá»ƒm cá»¥ thá»ƒ.
- Má»™t sá»‘ vÄƒn báº£n thá»§ cÃ´ng khÃ´ng cÃ³ `source_url`.
- Má»™t sá»‘ vÄƒn báº£n sá»­a Ä‘á»•i/phá»©c táº¡p cÃ³ thá»ƒ váº«n cÃ²n lá»—i tÃ¡ch Ä‘iá»u hoáº·c nháº­p ná»™i dung vÃ o chunk lÃ¢n cáº­n.
- BM25 index vÃ  Qdrant collection pháº£i cÃ¹ng phiÃªn báº£n dá»¯ liá»‡u; khÃ´ng nÃªn dÃ¹ng BM25 cá»§a collection khÃ¡c.
- CÃ¡c file `submission_v*.json`, `judge_cache*.json`, `scoring_*`, `answer_cache*` lÃ  artifact thÃ­ nghiá»‡m, khÃ´ng pháº£i dá»¯ liá»‡u nguá»“n chÃ­nh.

Khi dÃ¹ng dá»¯ liá»‡u cho bÃ i toÃ¡n phÃ¡p lÃ½ cÃ³ yÃªu cáº§u hiá»‡u lá»±c theo thá»i Ä‘iá»ƒm, cáº§n bá»• sung Ä‘á»“ thá»‹ quan há»‡ sá»­a Ä‘á»•i/thay tháº¿/háº¿t hiá»‡u lá»±c thay vÃ¬ chá»‰ dá»±a vÃ o `ngay_ban_hanh` hoáº·c `nam`.

## 13. Danh má»¥c bÃ n giao

### 13.1. Báº¯t buá»™c

| ÄÆ°á»ng dáº«n | Má»¥c Ä‘Ã­ch |
|---|---|
| `data/corpus_vbpl_v2_parsefix_20260628/articles.parquet` | Dá»¯ liá»‡u cáº¥p Ä‘iá»u/chunk |
| `data/corpus_vbpl_v2_parsefix_20260628/documents.parquet` | Metadata cáº¥p vÄƒn báº£n |
| `data/corpus_vbpl_v2_parsefix_20260628/manifest.json` | Provenance vÃ  danh má»¥c vÄƒn báº£n |
| `data/corpus_vbpl_v2_parsefix_20260628/inventory.md` | Thá»‘ng kÃª corpus |
| `data/corpus_vbpl_v2_parsefix_20260628/README.md` | MÃ´ táº£ corpus |
| `data/corpus_vbpl_v2_parsefix_20260628/PARSEFIX_NOTES.md` | Ghi chÃº sá»­a lá»—i parse |
| `data/bm25_vbpl_aiteam_meta_parsefix_20260628.pkl` | BM25 index |
| `data/manual_vbpl/` | VÄƒn báº£n bá»• sung thá»§ cÃ´ng |
| `docs/DATA_DESCRIPTION.md` | TÃ i liá»‡u mÃ´ táº£ dá»¯ liá»‡u |

### 13.2. Khuyáº¿n nghá»‹ kÃ¨m theo Ä‘á»ƒ tÃ¡i láº­p

| ÄÆ°á»ng dáº«n | Má»¥c Ä‘Ã­ch |
|---|---|
| `docs/QDRANT_DATA_METADATA_2026_06_28.md` | Audit Qdrant vÃ  metadata |
| `docs/PARSE_CHUNK_FIX_2026_06_28.md` | Chi tiáº¿t sá»­a lá»—i parse/chunk |
| `scripts/build_v72_metadata_arch.py` | Pipeline retrieval metadata-aware |
| `scripts/build_v75_prompt_selector.py` | Pipeline selector LLM |
| `scripts/build_v75_v72_selector_repair.py` | Pipeline má»›i nháº¥t |
| `ingest/run_vbpl.py` | Ingest dá»¯ liá»‡u |
| `ingest/manual_docs.py` | Náº¡p vÄƒn báº£n thá»§ cÃ´ng |
| `ingest/parse_vbpl.py` | Parse vÄƒn báº£n |
| `ingest/chunk.py` | TÃ¡ch chunk |

## 14. LiÃªn há»‡ vÃ  cáº­p nháº­t

Khi thay Ä‘á»•i collection Qdrant, BM25 index hoáº·c corpus export, cáº§n cáº­p nháº­t láº¡i cÃ¡c má»¥c sau trong tÃ i liá»‡u:

- TÃªn collection.
- TÃªn file BM25.
- Sá»‘ lÆ°á»£ng vÄƒn báº£n/chunk/point.
- Danh sÃ¡ch file bÃ n giao.
- Ghi chÃº cháº¥t lÆ°á»£ng vÃ  cÃ¡c lá»—i parse Ä‘Ã£ sá»­a.

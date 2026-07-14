# Retrieval Audit 2026-06-25

Muc tieu: dung vai tro "LLM giam sat retrieval", doc tung cau hoi va top chunk tra ve de xem retrieval co du can cu phap ly hay khong. Khong dung dap an BTC vi khong co ground truth that.

## Cau hinh da chay

- Qdrant: `http://localhost:6333`
- Collection: `vbpl_aiteam`
- Points: `285020`
- Embedding: `AITeamVN/Vietnamese_Embedding_v2`
- Reranker: `AITeamVN/Vietnamese_Reranker`
- Hybrid: dense + BM25
- Submission doi chieu: `data/submission_v50_full.json`
- Cau hoi: `C:/Users/PHONG/Downloads/R2AIStage1DATA.json`

## Nhan dinh nhanh

Index khong bi thieu toan cuc. Nhieu cau hoi retrieve du dieu dung o top 1-3, nen tran recall khong nam hoan toan o embedding/index.

Bottleneck lon hon nam o:

1. Version/current-law ranking: van ban cu van len cao hon van ban hien hanh hoac van ban moi.
2. Multi-hop retrieval: cau hoi gom 2-3 y thi top chunk co the chi bao phu 1 y.
3. Selector sau retrieval: co chunk dung trong top 8-12 nhung v50 van chon thieu hoac chon ban cu.
4. Mot so tai lieu parse ngan can kiem tra lai, dac biet `122/2021/ND-CP`, `17/2022/ND-CP`, va co the `70/2025/ND-CP`.

## Mau audit

### ID 1001

Cau hoi ve chuyen doi loai hinh doanh nghiep, cong viec ke toan, quyen doan kiem tra ke toan.

Top retrieval:

- `88/2015/QH13` Dieu 47: dung ve cong viec ke toan khi chuyen doi.
- `88/2015/QH13` Dieu 37: dung ve quyen va trach nhiem doan kiem tra ke toan.
- `99/2025/TT-BTC` Dieu 22: dung ve bao cao tai chinh khi chuyen doi.

Ket luan: retrieval du. Chatbot/pipeline nen giu ca Dieu 47 va Dieu 37; neu chi lay Dieu 47 hoac them noise cu thi la loi selector.

### ID 1016

Cau hoi ve khieu nai thue, quyen yeu cau ho so, hoan tra tien thu sai, tham quyen xu ly khieu nai.

Top retrieval:

- `108/2025/QH15` Dieu 42: dung va rat manh.
- `38/2019/QH14` Dieu 149: noi dung tuong duong/ban cu.
- Cac thong tu cu `28/2011`, `156/2013` len cao cho y tham quyen.

Ket luan: retrieval co can cu chinh, nhung co nhieu ban cu. Can version suppress de tranh lay cac van ban thue cu khi da co `108/2025/QH15`.

### ID 1053

Cau hoi gom 3 y: giai the con no thue, ho so hoan thue cham xu ly, gia han nop thue.

Top retrieval:

- `80/2021/TT-BTC` Dieu 27: dung mot phan ve xu ly ho so hoan thue.
- `80/2021/TT-BTC` Dieu 34: lien quan giai quyet ho so hoan thue.
- `19/2021/TT-BTC` Dieu 31: lien quan giao dich dien tu ve ho so xu ly no/gia han.
- `108/2025/QH15` Dieu 21 va `38/2019/QH14` Dieu 85: xoa no/phuc vu mot phan, nhung khong du.

Ket luan: retrieval thieu/yeu cho cau multi-hop. Can query decomposition theo tung y: `giai the con no thue`, `co quan thue cham hoan thue`, `gia han nop thue tiep nhan ho so`.

### ID 1055

Cau hoi ve ho so gia han nop thue do kho khan nganh nghe suy thoai va hoa hoan.

Top retrieval:

- `80/2021/TT-BTC` Dieu 24: dung top 1.
- `38/2019/QH14` Dieu 64: dung luat goc nhung chi rank 5.
- `126/2020/ND-CP` Dieu 19: lien quan nhung rank 7.
- Nhieu thong tu cu `156/2013`, `28/2011` chen cao.

Ket luan: retrieval du toi thieu, nhung version/currentness chua tot. Nen tang diem cho luat/nghi dinh/thong tu hien hanh cung ho van ban, ha manh ban cu.

### ID 1251

Cau hoi don gian: co phai thong bao ket qua thu viec khong.

Top retrieval:

- `45/2019/QH14` Dieu 27: dung top 1.
- Nhieu van ban huong dan/xu phat cu va phu tro chen sau.

Ket luan: retrieval tot. Khong nen them expansion/postprocess cho cau don gian vi de them noise.

### ID 1795

Cau hoi ve thanh vien la to chuc thay doi nguoi dai dien theo uy quyen va thu tuc thong bao co quan dang ky kinh doanh.

Top retrieval:

- `43/2010/ND-CP` Dieu 42: sai/qua cu, len top 1.
- `60/2005/QH11` Dieu 48: dung chu de nhung rat cu, len top 2.
- `168/2025/ND-CP` Dieu 54: dung thu tuc, top 3.
- `59/2020/QH14` Dieu 14: dung dieu kien nguoi dai dien, nhung chi top 8.
- V50 chon `68/2014/QH13` Dieu 15: dung chu de nhung khong phai ban hien hanh.

Ket luan: day la loi ranking/versioning ro rang. Index co chunk dung, nhung reranker de van ban cu va sai pham vi dung tren. Can successor map: `60/2005`, `68/2014` -> `59/2020`; `43/2010`, `01/2021` -> uu tien `168/2025` cho dang ky doanh nghiep.

### ID 1860

Cau hoi ve ho tro DNNVV khoi nghiep sang tao va tai lieu de duoc lua chon ho tro.

Top retrieval:

- `04/2017/QH14` Dieu 17: dung dieu kien/noi dung ho tro.
- `06/2022/TT-BKHDT` Dieu 14: dung tai lieu lua chon ho tro.
- `80/2021/ND-CP` Dieu 22: dung noi dung ho tro chi tiet.

Ket luan: retrieval du, nhung selector can giu it nhat 2-3 dieu. V50 chi co `04/2017/QH14` Dieu 17 la thieu y "tai lieu lua chon".

### ID 1999

Cau hoi ve cham dong kinh phi cong doan va khai thieu thue.

Top retrieval:

- `125/2020/ND-CP` Dieu 16: dung ve khai sai thieu thue.
- `12/2022/ND-CP` Dieu 38: dung ve kinh phi cong doan.
- `12/2022/ND-CP` Dieu 4: dung ve bien phap khac phuc hau qua chung.
- Co noise ban cu `28/2020/ND-CP`, `166/2013/TT-BTC`.

Ket luan: retrieval du, nhung can suppress van ban cu tuong duong de tang precision.

## Huong improve uu tien

1. Them current-version rerank layer truoc selector, khong chi postprocess submission:
   - `60/2005/QH11`, `68/2014/QH13` bi ha khi co `59/2020/QH14`.
   - `43/2010/ND-CP`, `01/2021/ND-CP` bi ha khi cau hoi dang ky doanh nghiep va co `168/2025/ND-CP`.
   - `28/2020/ND-CP` bi ha khi co `12/2022/ND-CP`.
   - Cac van ban quan ly thue cu bi ha khi co `38/2019/QH14` hoac `108/2025/QH15` phu hop.
2. Query decomposition cho cau multi-hop:
   - Tach cau hoi theo cac menh de phap ly.
   - Retrieve moi subquery.
   - Union theo dieu va score theo coverage so y.
3. Selector phai bao phu y:
   - Neu cau hoi co "va", "dong thoi", "neu ... thi ..." thi khong cat qua som top 1.
   - Uu tien giu cac dieu khac nhau bao phu cac cum keyword khac nhau.
4. Kiem tra lai data parse-short:
   - `122/2021/ND-CP` chi thay khoang 10 dieu trong index.
   - `17/2022/ND-CP` chi thay khoang 30 dieu.
   - `70/2025/ND-CP` chi thay khoang 3 dieu.

## Ket luan

Can cai thien retrieval/index theo huong ranking co hieu biet phien ban va coverage theo y, khong phai tiep tuc query expansion rong. Expansion rong da lam v56/v57 them noise va diem giam. Huong co kha nang tang that la sua ranking tai retrieval va sua selector bao phu multi-hop.

import zipfile
import requests
from pathlib import Path

API_KEY = "d850ae89d5815492bab4a95768c755b728b63b38"
RECEIPT_NO = "20190401004781"  # 실제 접수번호로 교체
OUT_ZIP = Path(f"document_{RECEIPT_NO}.zip")

params = {"crtfc_key": API_KEY, "rcept_no": RECEIPT_NO}
resp = requests.get("https://opendart.fss.or.kr/api/document.xml", params=params, timeout=60)
resp.raise_for_status()

content_type = resp.headers.get("Content-Type", "").lower()
if "json" in content_type:
    print("DART API returned JSON:")
    print(resp.json())
    raise SystemExit("문서가 제공되지 않았습니다. 접수번호나 인증키를 확인하세요.")

if resp.content.startswith(b"<?xml"):
    text = resp.content.decode("utf-8", errors="ignore")
    print("DART API returned XML response:")
    print(text)
    raise SystemExit("문서가 제공되지 않았습니다. 접수번호나 권한을 확인하세요.")

OUT_ZIP.write_bytes(resp.content)

with zipfile.ZipFile(OUT_ZIP) as zf:
    zf.extractall(OUT_ZIP.stem)
print(f"Saved and extracted to {OUT_ZIP.stem}")
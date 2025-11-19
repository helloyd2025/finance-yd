"""
OpenDART API 연계를 통해 원본 데이터 스키마를 구성하는 헬퍼 모듈.

지원 범위
---------
- DS001: 공시정보 (기업개황, 공시검색)
- DS002: 정기보고서 주요정보 (직원 현황 등 제한적 활용)
- DS003: 정기보고서 재무정보 (단일회사 전체 재무제표 등)

아직 DART에서 제공하지 않는 신용/공공정보, 내부 라벨 등은 None으로 채우고
추후 외부 데이터 소스와 결합해야 합니다.
"""

from __future__ import annotations

import dataclasses
import io
import logging
import zipfile
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple
from xml.etree import ElementTree

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

OPEN_DART_API_KEY = "d850ae89d5815492bab4a95768c755b728b63b38"
BASE_URL = "https://opendart.fss.or.kr/api"


# ---------------------------------------------------------------------------
# 원본 스키마 정의
# ---------------------------------------------------------------------------

ORIGINAL_COLUMNS: List[str] = [
    "기준년월", "업종(중분류)", "외감구분", "설립일자", "종업원수", "주소지시군구",
    "상장일자", "상장폐지일자", "유동자산", "비유동자산", "당좌자산", "재고자산", "유형자산",
    "재공품", "현금", "현금등가물", "상품유가증권", "현금성자산", "매출채권", "매출채권(전기)",
    "매출채권처분손실(당기)", "무형자산", "투자자산", "자산총계", "자산총계(전기)", "유동부채",
    "단기차입금", "차입금", "매입채무", "비유동부채", "부채총계", "자기자본(납입자본금)", "자본잉여금",
    "납입자본", "이익잉여금", "자본조정", "기타포괄손익누계액", "유보금", "자본총계", "전기자본총계",
    "매출액", "전기매출액", "매출원가", "매출총이익", "판매비와관리비", "법인세비용차감전순이익",
    "전기법인세차감전순이익", "법인세", "계속사업이익", "중단산업손익", "금융비용", "영업손익",
    "전기영업이익", "영업외수익", "영업외비용", "법인세차감전순이익", "당기순이익", "당기순이익(전기)",
    "현금흐름", "영업활동현금흐름", "투자활동현금흐름", "재무활동현금흐름", "부채상환계수",
    "영업이익이자보상배율", "이자비용", "사채이자(당기)", "이자보상배율", "적립금비율", "EBIT",
    "EBITDA", "청산가치율", "청산가치", "순운전자본", "순차입금", "재무비율_총자산증가율",
    "재무비율_부채비율", "재무비율_자기자본비율", "재무비율_유동비율", "재무비율_차입금의존도",
    "재무비율_매출액증가율", "재무비율_영업이익율", "재무비율_당기순이익율", "재무비율_매출원가율",
    "재무비율_판관비율", "재무비율_자기자본이익률(ROE)", "재무비율_매출채권회전율",
    "재무비율_재고자산회전율", "재무비율_매입채무회전율", "재무비율_총자산회전율", "재무비율_총자산순이익률",
    "재무비율_유동자산증가율", "재무비율_유형자산증가율", "단기차입금의존도", "당좌비율", "순차입금비율",
    "순운전자본회전율", "총자본회전율", "자기자본순이익율", "매출총이익율", "EBITDA마진율", "영업이익증가율",
    "당기순이익증가율", "EBITDA증가율", "OCF/매출액비용", "부채상환계수.1", "차입금/EBITDA",
    "EBITDA/금융비용", "사업장소유여부", "소유건축물건수", "소유건축물실거래가합계", "사업장권리침해여부",
    "소유건축물권리침해여부", "기업신용공여연체과목수(일보)(미해제)",
    "기업신용공여연체과목수(일보)(3개월내유지)(해제포함)",
    "기업신용공여연체과목수(일보)(6개월내유지)(해제포함)",
    "기업신용공여연체과목수(일보)(1년내유지)(해제포함)",
    "기업신용공여연체과목수(일보)(3년내유지)(해제포함)",
    "기업신용공여연체과목수(일보)(3개월내발생)(해제포함)",
    "기업신용공여연체과목수(일보)(6개월내발생)(해제포함)",
    "기업신용공여연체과목수(일보)(1년내발생)(해제포함)",
    "기업신용공여연체과목수(일보)(3년내발생)(해제포함)",
    "기업신용공여연체과목수(일보)(3개월내유지)(연체일수30일이상)(해제포함)",
    "기업신용공여연체과목수(일보)(6개월내유지)(연체일수30일이상)(해제포함)",
    "기업신용공여연체과목수(일보)(1년내유지)(연체일수30일이상)(해제포함)",
    "기업신용공여연체과목수(일보)(3년내유지)(연체일수30일이상)(해제포함)",
    "기업신용공여30일이상연체과목수(일보)(해제포함)",
    "기업신용공여30일이상연체과목수(일보)(미해제)",
    "기업신용공여30일이상연체과목수(일보)(이자연체)(해제포함)",
    "기업신용공여30일이상연체과목수(일보)(이자연체)(미해제)",
    "기업신용공여연체기관수(일보)(미해제)",
    "기업신용공여30일이상연체기관수(일보)(해제포함)",
    "기업신용공여30일이상연체기관수(일보)(미해제)",
    "기업신용공여30일이상연체기관수(일보)(연체)(해제포함)",
    "기업신용공여30일이상연체기관수(일보)(이자연체)(미해제)",
    "기업신용공여연체기관수(일보)(3개월내유지)(연체일수30일이상)(해제포함)",
    "기업신용공여연체기관수(일보)(6개월내유지)(연체일수30일이상)(해제포함)",
    "기업신용공여연체기관수(일보)(1년내유지)(연체일수30일이상)(해제포함)",
    "기업신용공여연체기관수(일보)(3년내유지)(연체일수30일이상)(해제포함)",
    "기업신용공여연체최장연체일수(일보)(3개월내유지)(해제포함)",
    "기업신용공여연체최장연체일수(일보)(6개월내유지)(해제포함)",
    "기업신용공여연체최장연체일수(일보)(1년내유지)(해제포함)",
    "기업신용공여연체최장연체일수(일보)(3년내유지)(해제포함)",
    "기업신용공여연체최장연체일수(일보)(5년내유지)(해제포함)",
    "기업신용공여연체최장연체일수(일보)(3개월내발생)(해제포함)",
    "기업신용공여연체최장연체일수(일보)(6개월내발생)(해제포함)",
    "기업신용공여연체최장연체일수(일보)(1년내발생)(해제포함)",
    "기업신용공여연체최장연체일수(일보)(3년내발생)(해제포함)",
    "기업신용공여연체최장연체일수(일보)(5년내발생)(해제포함)",
    "신용도판단공공정보건수(CIS)(5년내발생)(해제포함)",
    "신용도판단정보공공정보건수(CIS)(미해제)",
    "신용도판단정보공공정보건수(관련인제외)(CIS)(당월유지)(해제포함)",
    "공공정보(국세,지방세,관세체납)건수(CIS)(미해제)",
    "공공정보(국세,지방세,관세체납)건수(CIS)(5년내발생)",
    "공공정보(국세,지방세,관세체납,고용산재체납)건수(CIS)(미해제)",
    "공공정보(국세,지방세,관세체납,고용산재체납)건수(CIS)(5년내발생)",
    "신용도판단정보공공정보최근발생일자로부터경과일수(CIS)(해제,삭제)",
    "신용도판단정보공공정보최근해제일자로부터경과일수(CIS)(해제,삭제)",
    "기업신용평가등급(구간화)", "모형개발용Performance(향후1년내부도여부)",
]


# ---------------------------------------------------------------------------
# DS003: 재무 계정 → 원본 컬럼 맵핑
# ---------------------------------------------------------------------------

ACCOUNT_NAME_MAP: Dict[str, str] = {
    # 재무상태표
    "유동자산": "유동자산",
    "비유동자산": "비유동자산",
    "당좌자산": "당좌자산",
    "재고자산": "재고자산",
    "유형자산": "유형자산",
    "재공품": "재공품",
    "현금및현금성자산": "현금성자산",
    "현금및예치금": "현금",
    "현금": "현금",
    "매출채권": "매출채권",
    "전기매출채권": "매출채권(전기)",
    "무형자산": "무형자산",
    "투자자산": "투자자산",
    "자산총계": "자산총계",
    "전기자산총계": "자산총계(전기)",
    "유동부채": "유동부채",
    "비유동부채": "비유동부채",
    "부채총계": "부채총계",
    "자본총계": "자본총계",
    "전기자본총계": "전기자본총계",
    "납입자본": "납입자본",
    "자본잉여금": "자본잉여금",
    "이익잉여금": "이익잉여금",
    "기타포괄손익누계액": "기타포괄손익누계액",
    "매입채무": "매입채무",
    "단기차입금": "단기차입금",
    "차입금": "차입금",
    # 손익계산서
    "매출액": "매출액",
    "매출원가": "매출원가",
    "매출총이익": "매출총이익",
    "판매비와관리비": "판매비와관리비",
    "영업이익": "영업손익",
    "영업손익": "영업손익",
    "법인세비용차감전계속사업이익": "법인세비용차감전순이익",
    "법인세비용차감전순이익": "법인세비용차감전순이익",
    "당기순이익": "당기순이익",
    "전기당기순이익": "당기순이익(전기)",
    "법인세비용": "법인세",
    "금융비용": "금융비용",
    "영업외수익": "영업외수익",
    "영업외비용": "영업외비용",
    "계속영업이익": "계속사업이익",
    "중단사업손익": "중단산업손익",
    "전기영업이익": "전기영업이익",
    # 현금흐름표
    "영업활동현금흐름": "영업활동현금흐름",
    "투자활동현금흐름": "투자활동현금흐름",
    "재무활동현금흐름": "재무활동현금흐름",
    "기말현금및현금성자산": "현금흐름",
    # 성과지표
    "EBIT": "EBIT",
    "EBITDA": "EBITDA",
}

ACCOUNT_ID_MAP: Dict[str, str] = {
    "ifrs_Cash": "현금",
    "ifrs_CashAndCashEquivalents": "현금성자산",
    "ifrs_CurrentAssets": "유동자산",
    "ifrs_NoncurrentAssets": "비유동자산",
    "ifrs_CurrentLiabilities": "유동부채",
    "ifrs_NoncurrentLiabilities": "비유동부채",
    "ifrs_Liabilities": "부채총계",
    "ifrs_Equity": "자본총계",
    "ifrs_Revenue": "매출액",
    "ifrs_CostOfSales": "매출원가",
    "ifrs_GrossProfit": "매출총이익",
    "ifrs_ProfitLossFromOperatingActivities": "영업손익",
    "ifrs_ProfitLossBeforeTax": "법인세비용차감전순이익",
    "ifrs_ProfitLoss": "당기순이익",
    "ifrs_CashFlowsFromUsedInOperatingActivities": "영업활동현금흐름",
    "ifrs_CashFlowsFromUsedInInvestingActivities": "투자활동현금흐름",
    "ifrs_CashFlowsFromUsedInFinancingActivities": "재무활동현금흐름",
}

# NOTE: ACCOUNT_NAME_MAP는 최소 매핑만 채워두고, 실제 사용 시 추가 계정명을 보완해야 한다.
# IFRS 한글 표준 계정명은 기업별로 약간씩 다르므로 사전 확인이 필요하다.


# ---------------------------------------------------------------------------
# 데이터 클래스
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class FinancialSnapshot:
    corp_code: str
    corp_name: str
    bsns_year: int
    reprt_code: str
    data: Dict[str, Optional[str]]


# ---------------------------------------------------------------------------
# OpenDART API 클라이언트
# ---------------------------------------------------------------------------

class OpenDartClient:
    def __init__(self, api_key: str = OPEN_DART_API_KEY, session: Optional[requests.Session] = None):
        self.api_key = api_key
        self.session = session or requests.Session()

    def _request(self, path: str, params: Dict[str, str]) -> Dict:
        params = {"crtfc_key": self.api_key, **params}
        url = f"{BASE_URL}/{path}"
        LOGGER.debug("Requesting %s with params=%s", url, params)
        resp = self.session.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "")
        if status not in ("000", "", "013"):  # 013: 조회 데이터 없음
            raise RuntimeError(f"DART API error {status}: {data.get('message')}")
        return data

    # ------------------------------------------------------------------
    # DS001
    # ------------------------------------------------------------------

    def get_corp_code_zip(self) -> pd.DataFrame:
        """고유번호 목록(zip/xml)을 DataFrame으로 반환."""
        url = f"{BASE_URL}/corpCode.xml"
        resp = self.session.get(url, params={"crtfc_key": self.api_key}, timeout=30)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            name = zf.namelist()[0]
            with zf.open(name) as fp:
                xml_text = fp.read()
        root = ElementTree.fromstring(xml_text)
        rows = []
        for corp in root.findall("list"):
            rows.append({
                "corp_code": corp.findtext("corp_code"),
                "corp_name": corp.findtext("corp_name"),
                "stock_code": corp.findtext("stock_code"),
                "modify_date": corp.findtext("modify_date"),
            })
        return pd.DataFrame(rows)

    def get_corp_code(self, corp_name: str, corp_codes: Optional[pd.DataFrame] = None) -> Optional[str]:
        if corp_codes is None:
            corp_codes = self.get_corp_code_zip()
        matches = corp_codes.loc[corp_codes["corp_name"] == corp_name]
        if matches.empty:
            LOGGER.warning("Corp code not found for %s", corp_name)
            return None
        return matches.iloc[0]["corp_code"]

    def get_company_overview(self, corp_code: str) -> Dict[str, str]:
        data = self._request("company.json", {"corp_code": corp_code})
        return data or {}

    def search_reports(self, corp_code: str, bgn_de: str, end_de: str, last_reprt_at: str = "Y") -> List[Dict[str, str]]:
        params = {
            "corp_code": corp_code,
            "bgn_de": bgn_de,
            "end_de": end_de,
            "last_reprt_at": last_reprt_at,
        }
        data = self._request("list.json", params)
        return data.get("list", [])

    # ------------------------------------------------------------------
    # DS002
    # ------------------------------------------------------------------

    def get_employee_status(self, corp_code: str, bsns_year: int, reprt_code: str) -> List[Dict[str, str]]:
        try:
            data = self._request("empSttus.json", {
                "corp_code": corp_code,
                "bsns_year": str(bsns_year),
                "reprt_code": reprt_code,
            })
        except RuntimeError as exc:
            LOGGER.info("Employee status not available: %s", exc)
            return []
        return data.get("list", [])

    # ------------------------------------------------------------------
    # DS003
    # ------------------------------------------------------------------

    def get_single_financial_statements(
        self,
        corp_code: str,
        bsns_year: int,
        reprt_code: str,
    ) -> List[Dict[str, str]]:
        data = self._request(
            "fnlttSinglAcntAll.json",
            {
                "corp_code": corp_code,
                "bsns_year": str(bsns_year),
                "reprt_code": reprt_code,
                "fs_div": "CFS",
            },
        )
        return data.get("list", [])


# ---------------------------------------------------------------------------
# 매핑 유틸리티
# ---------------------------------------------------------------------------

def normalize_account_name(account_name: str) -> str:
    name = account_name.replace(" ", "")
    for char in (" ", "(", ")"):
        name = name.replace(char, "")
    return name


def map_financial_accounts(records: Iterable[Dict[str, str]]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for item in records:
        account_nm = item.get("account_nm")
        account_id = item.get("account_id")
        thstrm = item.get("thstrm_amount")
        if not account_nm or thstrm in (None, "", "-"):
            continue
        normalized = normalize_account_name(account_nm)
        target_col = ACCOUNT_NAME_MAP.get(normalized)
        if not target_col and account_id:
            target_col = ACCOUNT_ID_MAP.get(account_id)
        if not target_col:
            continue
        try:
            amount = float(thstrm.replace(",", ""))
        except ValueError:
            continue
        values[target_col] = amount
    return values


def extract_employee_count(emp_status: List[Dict[str, str]]) -> Optional[int]:
    if not emp_status:
        return None
    preferred = [
        item for item in emp_status
        if item.get("sm") == "합계" or item.get("se") == "합계" or item.get("se") == "전체"
    ]
    sexdstn_total = [
        item for item in emp_status
        if str(item.get("sexdstn")).strip() == "합계" or str(item.get("fo_bbm")).strip() == "성별합계"
    ]
    candidates = preferred or sexdstn_total or emp_status
    numeric_keys = (
        "janng_totl_nmpr",
        "totl_nmpr",
        "janng_nmpr",
        "totl",
        "totl_emply_co",
        "emply_co",
        "ppl_num",
        "sm",
    )

    def _parse_int(value: object) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return int(value)
        text = str(value).strip()
        if not text:
            return None
        cleaned = "".join(ch for ch in text if ch.isdigit() or ch in ("-", "."))
        if cleaned in ("", "-", ".", "-."):
            return None
        try:
            return int(float(cleaned))
        except ValueError:
            return None

    for item in candidates:
        for key in numeric_keys:
            raw = item.get(key)
            parsed = _parse_int(raw)
            if parsed is not None:
                return parsed
    return None


# ---------------------------------------------------------------------------
# 파생 지표 계산
# ---------------------------------------------------------------------------

def _to_float(value: Optional[object]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_div(numerator: Optional[object], denominator: Optional[object]) -> Optional[float]:
    num = _to_float(numerator)
    den = _to_float(denominator)
    if num is None or den in (None, 0):
        return None
    return num / den


def _safe_sub(a: Optional[object], b: Optional[object]) -> Optional[float]:
    va = _to_float(a)
    vb = _to_float(b)
    if va is None or vb is None:
        return None
    return va - vb


def _average(*values: Optional[object]) -> Optional[float]:
    nums = [v for v in (_to_float(val) for val in values) if v is not None]
    if not nums:
        return None
    return sum(nums) / len(nums)


def compute_derived_metrics(row: Dict[str, Optional[object]]) -> None:
    """
    원본 수치 기반으로 재무 비율 및 파생 지표를 계산해 row에 채웁니다.
    계산 불가 시 해당 컬럼은 그대로 두거나 None으로 유지합니다.
    """

    total_assets = _to_float(row.get("자산총계"))
    prev_total_assets = _to_float(row.get("자산총계(전기)"))
    total_liabilities = _to_float(row.get("부채총계"))
    total_equity = _to_float(row.get("자본총계"))
    prev_total_equity = _to_float(row.get("전기자본총계"))
    current_assets = _to_float(row.get("유동자산"))
    current_liabilities = _to_float(row.get("유동부채"))
    inventory = _to_float(row.get("재고자산"))
    cash_equivalents = _to_float(row.get("현금성자산")) or _to_float(row.get("현금"))
    borrowings = _to_float(row.get("차입금"))
    short_borrowings = _to_float(row.get("단기차입금"))
    sales = _to_float(row.get("매출액"))
    prev_sales = _to_float(row.get("전기매출액"))
    operating_income = _to_float(row.get("영업손익")) or _to_float(row.get("영업이익"))
    prev_operating_income = _to_float(row.get("전기영업이익"))
    net_income = _to_float(row.get("당기순이익"))
    prev_net_income = _to_float(row.get("당기순이익(전기)"))
    cost_of_sales = _to_float(row.get("매출원가"))
    sga_expense = _to_float(row.get("판매비와관리비"))
    accounts_receivable = _to_float(row.get("매출채권"))
    accounts_payable = _to_float(row.get("매입채무"))
    depreciation = _to_float(row.get("감가상각비")) or 0.0
    amortization = _to_float(row.get("무형자산상각비")) or 0.0
    interest_expense = _to_float(row.get("이자비용"))
    financial_cost = _to_float(row.get("금융비용"))
    tax_expense = _to_float(row.get("법인세")) or _to_float(row.get("법인세비용"))

    # EBIT 계산 우선순위: 기존 값 → 영업이익 → 순이익 기반 복원
    if (row.get("EBIT") in (None, "", "NaN")):
        if operating_income is not None:
            row["EBIT"] = operating_income
        elif net_income is not None and (
            interest_expense is not None or financial_cost is not None or tax_expense is not None
        ):
            ebit_from_net = net_income
            if interest_expense is not None:
                ebit_from_net += interest_expense
            if financial_cost is not None:
                ebit_from_net += financial_cost
            if tax_expense is not None:
                ebit_from_net += tax_expense
            row["EBIT"] = ebit_from_net

    ebit = _to_float(row.get("EBIT")) or operating_income

    # EBITDA 계산 우선순위: 기존 값 → EBIT 기반 → 순이익 기반
    if (row.get("EBITDA") in (None, "", "NaN")):
        if ebit is not None:
            row["EBITDA"] = ebit + (depreciation or 0.0) + (amortization or 0.0)
        elif net_income is not None:
            ebitda_from_net = net_income
            if interest_expense is not None:
                ebitda_from_net += interest_expense
            if financial_cost is not None:
                ebitda_from_net += financial_cost
            if tax_expense is not None:
                ebitda_from_net += tax_expense
            ebitda_from_net += (depreciation or 0.0) + (amortization or 0.0)
            row["EBITDA"] = ebitda_from_net

    ebitda = _to_float(row.get("EBITDA"))
    operating_cash_flow = _to_float(row.get("영업활동현금흐름"))
    total_cash_flow = _to_float(row.get("현금흐름"))
    gross_profit = _to_float(row.get("매출총이익"))

    # 성장률
    row["재무비율_총자산증가율"] = _safe_div(
        _safe_sub(total_assets, prev_total_assets),
        prev_total_assets,
    )
    row["재무비율_매출액증가율"] = _safe_div(
        _safe_sub(sales, prev_sales),
        prev_sales,
    )
    row["영업이익증가율"] = _safe_div(
        _safe_sub(operating_income, prev_operating_income),
        prev_operating_income,
    )
    row["당기순이익증가율"] = _safe_div(
        _safe_sub(net_income, prev_net_income),
        prev_net_income,
    )

    # 재무 구조
    row["재무비율_부채비율"] = _safe_div(total_liabilities, total_equity)
    row["재무비율_자기자본비율"] = _safe_div(total_equity, total_assets)
    row["재무비율_유동비율"] = _safe_div(current_assets, current_liabilities)
    row["재무비율_차입금의존도"] = _safe_div(borrowings, total_assets)
    row["단기차입금의존도"] = _safe_div(short_borrowings, borrowings)
    row["당좌비율"] = _safe_div(
        _safe_sub(current_assets, inventory),
        current_liabilities,
    )

    # 수익성
    avg_equity = _average(total_equity, prev_total_equity)
    row["재무비율_영업이익율"] = _safe_div(operating_income, sales)
    row["재무비율_당기순이익율"] = _safe_div(net_income, sales)
    row["재무비율_매출원가율"] = _safe_div(cost_of_sales, sales)
    row["재무비율_판관비율"] = _safe_div(sga_expense, sales)
    row["재무비율_총자산회전율"] = _safe_div(sales, total_assets)
    row["총자본회전율"] = row.get("재무비율_총자산회전율")
    row["재무비율_총자산순이익률"] = _safe_div(net_income, total_assets)
    row["재무비율_자기자본이익률(ROE)"] = _safe_div(net_income, avg_equity or total_equity)
    row["자기자본순이익율"] = row.get("재무비율_자기자본이익률(ROE)")
    row["매출총이익율"] = _safe_div(gross_profit, sales)
    ebitda_margin = _safe_div(ebitda, sales)
    row["EBITDA마진율"] = ebitda_margin * 100 if ebitda_margin is not None else None

    # 효율성
    row["재무비율_매출채권회전율"] = _safe_div(sales, accounts_receivable)
    row["재무비율_재고자산회전율"] = _safe_div(cost_of_sales, inventory)
    row["재무비율_매입채무회전율"] = _safe_div(cost_of_sales, accounts_payable)

    net_working_capital = _safe_sub(current_assets, current_liabilities)
    row["순운전자본"] = net_working_capital if net_working_capital is not None else row.get("순운전자본")
    row["재무비율_유동자산증가율"] = _safe_div(
        _safe_sub(current_assets, row.get("유동자산(전기)")),
        row.get("유동자산(전기)"),
    )
    row["재무비율_유형자산증가율"] = _safe_div(
        _safe_sub(row.get("유형자산"), row.get("유형자산(전기)")),
        row.get("유형자산(전기)"),
    )
    row["순운전자본회전율"] = _safe_div(sales, row.get("순운전자본"))

    net_debt = None
    if borrowings is not None:
        net_debt = borrowings - (cash_equivalents or 0.0)
    row["순차입금"] = net_debt if net_debt is not None else row.get("순차입금")
    row["순차입금비율"] = _safe_div(row.get("순차입금"), total_equity)

    # 현금흐름 기반
    row["OCF/매출액비용"] = _safe_div(operating_cash_flow, sales)
    row["부채상환계수"] = _safe_div(total_liabilities, operating_cash_flow)
    row["부채상환계수.1"] = _safe_div(borrowings, operating_cash_flow)

    # 이자보상 능력
    row["영업이익이자보상배율"] = _safe_div(operating_income, financial_cost or interest_expense)
    row["이자보상배율"] = _safe_div(ebit, interest_expense or financial_cost)
    row["EBITDA/금융비용"] = _safe_div(ebitda, financial_cost or interest_expense)
    row["차입금/EBITDA"] = _safe_div(borrowings, ebitda)

    # EBITDA 증감률은 이전 값이 없으면 계산하지 않음
    prev_ebitda = row.get("EBITDA(전기)")
    row["EBITDA증가율"] = _safe_div(
        _safe_sub(ebitda, prev_ebitda),
        prev_ebitda,
    )


def build_dataset_row(
    corp_name: str,
    bsns_year: int,
    reprt_code: str,
    api_key: str = OPEN_DART_API_KEY,
) -> FinancialSnapshot:
    client = OpenDartClient(api_key=api_key)
    corp_codes = client.get_corp_code_zip()
    corp_code = client.get_corp_code(corp_name, corp_codes)
    if not corp_code:
        raise ValueError(f"기업명을 찾을 수 없습니다: {corp_name}")

    overview = client.get_company_overview(corp_code)
    statements = client.get_single_financial_statements(corp_code, bsns_year, reprt_code)
    emp_status = client.get_employee_status(corp_code, bsns_year, reprt_code)
   

    mapped_financials = map_financial_accounts(statements)
    employee_count = extract_employee_count(emp_status)

    row = defaultdict(lambda: None)  # type: ignore

    # 기본 정보 매핑
    row["기준년월"] = overview.get("acc_mt") or f"{bsns_year}12"
    row["업종(중분류)"] = overview.get("induty_code")
    row["설립일자"] = overview.get("est_dt")
    row["상장일자"] = overview.get("list_dt")  # 제공되지 않으면 None 유지
    row["상장폐지일자"] = overview.get("delist_dt")
    row["주소지시군구"] = overview.get("adres")
    row["외감구분"] = overview.get("corp_cls")
    row["종업원수"] = employee_count

    # 재무제표 매핑
    for col, value in mapped_financials.items():
        row[col] = value

    # 사용할 수 없는 컬럼은 None으로 유지 (향후 외부 데이터 결합 필요)
    compute_derived_metrics(row)

    data = {col: row[col] for col in ORIGINAL_COLUMNS}
    return FinancialSnapshot(
        corp_code=corp_code,
        corp_name=corp_name,
        bsns_year=bsns_year,
        reprt_code=reprt_code,
        data=data,
    )


def build_dataframe(
    corp_names: List[str],
    bsns_year: int,
    reprt_code: str,
    api_key: str = OPEN_DART_API_KEY,
) -> pd.DataFrame:
    rows = []
    for corp in corp_names:
        try:
            snapshot = build_dataset_row(corp, bsns_year, reprt_code, api_key)
            rows.append(snapshot.data)
        except Exception as exc:
            LOGGER.error("Failed to build row for %s: %s", corp, exc)
    return pd.DataFrame(rows, columns=ORIGINAL_COLUMNS)


def export_to_csv(
    corp_names: List[str],
    bsns_year: int,
    reprt_code: str,
    out_path: str,
    api_key: str = OPEN_DART_API_KEY,
    encoding: str = "utf-8",
) -> pd.DataFrame:
    """
    지정한 기업들의 DART 데이터를 수집하여 CSV 파일로 저장합니다.

    Parameters
    ----------
    corp_names : List[str]
        수집 대상 기업명 목록.
    bsns_year : int
        사업연도(YYYY).
    reprt_code : str
        보고서 코드 (11011: 사업보고서, 11012: 반기보고서, 11013: 1분기보고서, 11014: 3분기보고서).
    out_path : str
        저장할 CSV 경로.
    api_key : str, optional
        OpenDART 인증키. 기본값은 설정된 글로벌 키.
    encoding : str, optional
        CSV 인코딩. 기본값은 utf-8.

    Returns
    -------
    pandas.DataFrame
        저장된 DataFrame을 반환합니다.
    """
    df = build_dataframe(corp_names, bsns_year, reprt_code, api_key=api_key)
    df.to_csv(out_path, index=False, encoding=encoding)
    LOGGER.info("Saved %d rows to %s", len(df), out_path)
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = export_to_csv(
        ["삼성전자"],
        bsns_year=2023,
        reprt_code="11011",
        out_path="dart_sample.csv",
    )
    print(sample.head())


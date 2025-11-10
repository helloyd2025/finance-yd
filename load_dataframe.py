"""
CSV 파일을 DataFrame으로 로드하는 스크립트
컬럼명에 쉼표가 포함된 경우를 처리합니다.
"""
import pandas as pd
import csv
import io
import re


def _make_unique_column_names(names):
    """중복 컬럼명에 .1, .2 ... 접미사를 붙여 고유화합니다."""
    seen = {}
    unique = []
    for name in names:
        base = name if name else "col"
        count = seen.get(base, 0)
        if count == 0 and base not in unique:
            unique.append(base)
            seen[base] = 1
        else:
            new_name = f"{base}.{count}"
            while new_name in seen or new_name in unique:
                count += 1
                new_name = f"{base}.{count}"
            unique.append(new_name)
            seen[base] = count + 1
            seen[new_name] = 1
    return unique


def _clean_column_tokens(tokens):
    """토큰 리스트의 공백/탭/따옴표/끝 쉼표를 정리합니다."""
    cleaned = []
    for t in tokens:
        name = (t.replace('\t', ' ')
                  .replace('"', '')
                  .strip()
                  .rstrip(',')
                  .strip())
        cleaned.append(name)
    return cleaned


def _parse_header_by_quotes_only(header_raw: str) -> list:
    """큰따옴표로 감싼 구간만을 컬럼으로 인식해 추출합니다. 콤마는 무시합니다."""
    # 연속된 "..." 구간들을 수집 (이중따옴표 이스케이프 처리 "" -> ")
    matches = re.findall(r'"((?:[^"]|"")*)"', header_raw)
    # 이스케이프 복원
    return [m.replace('""', '"') for m in matches]


def build_header_names_by_csv_rules(file_path: str, data_fields: int, prefer_quotes_only: bool = True):
    """원본 헤더를 파싱해 길이 보정 및 고유화한 컬럼명 리스트 생성.
    prefer_quotes_only=True이면 큰따옴표만으로 컬럼을 구분(콤마 무시).
    실패 시 표준 CSV 파서로 폴백.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        header_raw = f.readline().rstrip("\r\n")

    parsed = []
    if prefer_quotes_only:
        parsed = _parse_header_by_quotes_only(header_raw)
        # quotes-only 결과가 유효한 경우(최소 2개 이상) 그대로 사용
        if len(parsed) == data_fields:
            parsed = _clean_column_tokens(parsed)
        else:
            # 폴백: 표준 CSV 파싱
            parsed = next(csv.reader([header_raw], delimiter=",", quotechar='"', doublequote=True))
            parsed = _clean_column_tokens(parsed)
    else:
        parsed = next(csv.reader([header_raw], delimiter=",", quotechar='"', doublequote=True))
        parsed = _clean_column_tokens(parsed)

    # 길이 보정
    if data_fields and len(parsed) != data_fields:
        if len(parsed) > data_fields:
            parsed = parsed[:data_fields]
        else:
            parsed += [f"col_{i}" for i in range(len(parsed), data_fields)]
    return _make_unique_column_names(parsed)


def load_dataframe(file_path: str = "기업신용평가정보_합성데이터.csv", delimiter: str = None) -> pd.DataFrame:
    """
    CSV/TSV 파일을 pandas DataFrame으로 로드합니다.
    컬럼명에 쉼표나 줄바꿈이 포함된 경우를 올바르게 처리합니다.
    
    Args:
        file_path: CSV 파일 경로
        delimiter: 구분자 (None이면 자동 감지: 먼저 탭, 그 다음 쉼표)
        
    Returns:
        pandas DataFrame
    """
    print(f"CSV 파일 로드 중: {file_path}")
    
    # 구분자 자동 감지
    if delimiter is None:
        # 파일 첫 줄 읽어서 구분자 확인
        with open(file_path, 'r', encoding='utf-8') as f:
            first_line = f.read(1000)  # 첫 1000자만 읽기
            if '\t' in first_line:
                delimiter = '\t'
                print("  → 탭(\t) 구분자 감지")
            else:
                delimiter = ','
                print("  → 쉼표(,) 구분자 감지")
    
    # 방법 1: 탭 구분자 + 따옴표 처리
    # QUOTE_MINIMAL: 쉼표, 줄바꿈, 따옴표가 포함된 필드만 따옴표로 처리 (표준 방식)
    try:
        df = pd.read_csv(
            file_path, 
            encoding='utf-8',
            delimiter=delimiter,  # 구분자 명시
            quotechar='"',  # 따옴표 문자 명시
            quoting=csv.QUOTE_MINIMAL,  # 필요한 경우에만 따옴표 사용 (표준 CSV)
            skipinitialspace=True,  # 따옴표 뒤 공백 제거
            on_bad_lines='skip',  # 잘못된 줄 건너뛰기 (pandas 1.3+)
            keep_default_na=True,  # NaN 값 처리
            na_values=['', 'NA', 'N/A', 'null', 'NULL']  # 추가 NaN 값
        )
        print(f"데이터 로드 완료!")
        print(f"  - 행 수: {len(df)}")
        print(f"  - 열 수: {len(df.columns)}")
        
        # 컬럼명 정리 (앞뒤 공백, 탭, 줄바꿈, 따옴표 제거)
        df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\t', ' ').str.replace('"', '').str.strip()
        
        # 컬럼명에 쉼표나 줄바꿈이 포함된 경우 확인
        problematic_cols = [col for col in df.columns if (',' in col or '\n' in col or '\t' in col)]
        if problematic_cols:
            print(f"  ⚠️  경고: {len(problematic_cols)}개 컬럼에 특수문자가 포함되어 있습니다.")
            print(f"      예시: {problematic_cols[0][:50]}...")
        else:
            print(f"  ✓ 모든 컬럼명이 올바르게 파싱되었습니다.")
        
        print(f"  - 처음 5개 컬럼명:")
        for i, col in enumerate(list(df.columns)[:5]):
            print(f"      {i+1}. {col[:60]}...")
        return df
        
    except Exception as e:
        print(f"첫 번째 방법 실패: {e}")
        print("두 번째 방법 시도 중...")
        
        # 방법 2: 다른 인코딩 시도
        try:
            df = pd.read_csv(
                file_path,
                encoding='cp949',  # 한글 Windows 인코딩
                delimiter=delimiter,
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
                skipinitialspace=True
            )
            df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\t', ' ').str.replace('"', '').str.strip()
            print(f"데이터 로드 완료! (cp949 인코딩)")
            return df
        except Exception as e2:
            print(f"두 번째 방법도 실패: {e2}")
            print("세 번째 방법 시도 중...")
            
            # 방법 3: escapechar 옵션 추가 (이스케이프 문자 처리)
            try:
                df = pd.read_csv(
                    file_path,
                    encoding='utf-8',
                    delimiter=delimiter,
                    quotechar='"',
                    quoting=csv.QUOTE_MINIMAL,
                    escapechar='\\',  # 이스케이프 문자
                    skipinitialspace=True
                )
                df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\t', ' ').str.replace('"', '').str.strip()
                print(f"데이터 로드 완료! (escapechar 사용)")
                return df
            except Exception as e3:
                print(f"세 번째 방법도 실패: {e3}")
                print("네 번째 방법 시도 중...")
                
                # 방법 4: 기본 설정으로 재시도
                df = pd.read_csv(file_path, encoding='utf-8', delimiter=delimiter)
                df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\t', ' ').str.replace('"', '').str.strip()
                print(f"기본 설정으로 로드 완료 (경고: 컬럼 파싱 문제가 있을 수 있습니다)")
                return df


def load_dataframe_with_fallback(file_path: str = "기업신용평가정보_합성데이터.csv") -> pd.DataFrame:
    """
    CSV 파일을 로드하는데, 여러 방법을 시도하여 가장 좋은 결과를 반환합니다.
    BOM이나 인코딩 문제도 처리합니다.
    
    Args:
        file_path: CSV 파일 경로
        
    Returns:
        pandas DataFrame
    """
    print(f"CSV 파일 로드 중 (여러 방법 시도): {file_path}")
    
    # 방법 1: BOM 제거 후 읽기
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            # UTF-8 BOM 제거
            if content.startswith(b'\xef\xbb\xbf'):
                content = content[3:]
            text = content.decode('utf-8')
            # 구분자 자동 감지
            delimiter = '\t' if '\t' in text[:1000] else ','
            df = pd.read_csv(
                io.StringIO(text),
                delimiter=delimiter,
                quotechar='"',
                quoting=csv.QUOTE_MINIMAL,
                skipinitialspace=True
            )
            df.columns = df.columns.str.strip().str.replace('\n', ' ').str.replace('\t', ' ')
            print("✓ BOM 제거 후 로드 성공!")
            return df
    except Exception as e:
        print(f"BOM 제거 방법 실패: {e}")
    
    # 방법 2: 기본 load_dataframe 사용
    return load_dataframe(file_path)


def check_csv_columns(file_path: str = "기업신용평가정보_합성데이터.csv"):
    """
    CSV 파일의 컬럼 파싱 상태를 확인합니다.
    
    Args:
        file_path: CSV 파일 경로
    """
    print("="*60)
    print("CSV 컬럼 파싱 상태 확인")
    print("="*60)
    
    # 여러 방법으로 시도
    methods = [
        ("기본 방법", {"encoding": "utf-8"}),
        ("QUOTE_MINIMAL", {"encoding": "utf-8", "quotechar": '"', "quoting": csv.QUOTE_MINIMAL}),
        ("QUOTE_ALL", {"encoding": "utf-8", "quotechar": '"', "quoting": csv.QUOTE_ALL}),
       
    ]
    
    for method_name, options in methods:
        try:
            df = pd.read_csv(file_path, nrows=0, **options)  # 헤더만 읽기
            cols_with_comma = [col for col in df.columns if ',' in col]
            print(f"\n{method_name}:")
            print(f"  - 컬럼 수: {len(df.columns)}")
            if cols_with_comma:
                print(f"  - ⚠️  쉼표 포함 컬럼: {len(cols_with_comma)}개")
                print(f"      예시: {cols_with_comma[0]}")
            else:
                print(f"  - ✓ 모든 컬럼 정상")
        except Exception as e:
            print(f"\n{method_name}: 실패 - {e}")


def load_by_data_fields(file_path: str) -> pd.DataFrame:
    """
    헤더가 비표준으로 깨지는 경우, 첫 데이터 줄의 실제 필드 개수를 기준으로
    임시 컬럼명을 생성해 안정적으로 로드합니다.
    
    주의: 인덱스 매핑
    - 원본 CSV 파일에서 데이터 행은 행 2부터 시작합니다 (행 0: BOM?, 행 1: 헤더)
    - DataFrame 인덱스는 0부터 시작합니다
    - 원본 CSV 행 번호 = DataFrame 인덱스 + 2
    - 예: DataFrame 인덱스 179438 → 원본 CSV 행 179440
    
    Returns:
        pandas DataFrame (인덱스는 0부터 시작)
    """
    # 첫 데이터 줄에서 필드 수 계산
    with open(file_path, "r", encoding="utf-8") as f:
        _ = f.readline()  # 헤더 스킵
        first_data = f.readline().rstrip("\r\n")

    reader = csv.reader([first_data], delimiter=",", quotechar='"', doublequote=True)
    data_fields = len(next(reader, []))
    # 헤더를 CSV 규칙에 맞춰 파싱한 컬럼명 구성 (길이 보정 + 고유화 포함)
    names = build_header_names_by_csv_rules(file_path, data_fields)

    df = pd.read_csv(
        file_path,
        header=None,
        names=names,
        skiprows=1,              # 원본 헤더 스킵 (행 1)
        sep=",",
        engine="python",
        quotechar='"',
        doublequote=True,
        skipinitialspace=True,
        on_bad_lines="skip",
        usecols=range(data_fields)
    )
    # DataFrame 인덱스는 0부터 시작 (원본 CSV 행 2가 인덱스 0이 됨)
    return df


if __name__ == "__main__":
    # CSV 컬럼 파싱 상태 확인
    check_csv_columns("기업신용평가정보_합성데이터.csv")
    
    print("\n" + "="*60)
    print("CSV 파일 로드")
    print("="*60)
    
    # 헤더 문제 회피를 위해 데이터 줄 기준 로드 사용
    df = load_by_data_fields("기업신용평가정보_합성데이터.csv")

    # 마지막 컬럼이 예상 타겟명인지 확인하고, 아니면 헤더 재구성하여 적용
    expected_last = "모형개발용Performance(향후1년내부도여부)"
    if df.columns[-1] != expected_last:
        try:
            names = build_header_names_by_csv_rules("기업신용평가정보_합성데이터.csv", df.shape[1])
            df.columns = names
        except Exception:
            pass

    # 마지막 2개 컬럼 및 샘플 출력
    try:
        print("\n" + "="*60)
        print("마지막 컬럼 검증")
        print("="*60)
        print("마지막 2개 컬럼:", df.columns[-2:].tolist())
        print("앞 5행 마지막 2개 값:\n", df.iloc[:5, -2:])
    except Exception:
        pass
    
    # DataFrame 정보 출력
    print("\n" + "="*60)
    print("DataFrame 정보")
    print("="*60)
    print(df.info())
    print("\n" + "="*60)
    print("처음 5행 미리보기")
    print("="*60)
    print(df.head())
    
    # DataFrame이 변수 'df'에 저장되었습니다
    print("\n✓ DataFrame이 'df' 변수에 저장되었습니다!")


"""
트레이딩 계산기 CLI 인터페이스
사용자가 쉽게 거래를 입력하고 결과를 확인할 수 있는 인터페이스
"""

from trading_calculator import TradingCalculator
from datetime import datetime
import sys

def main():
    print("="*60)
    print("트레이딩 계산기")
    print("="*60)
    
    # 초기 자본금 입력
    try:
        initial_capital = float(input("초기 자본금을 입력하세요 (기본값: 8,000,000): ") or "8000000")
    except ValueError:
        print("잘못된 입력입니다. 기본값 8,000,000원을 사용합니다.")
        initial_capital = 8000000
    
    calculator = TradingCalculator(initial_capital=initial_capital)
    
    print(f"\n초기 자본금: {initial_capital:,.0f}원으로 계산기를 시작합니다.\n")
    
    while True:
        print("\n" + "-"*60)
        print("메뉴:")
        print("1. 거래 기록 추가")
        print("2. 현재 상태 확인")
        print("3. 엑셀 파일로 저장")
        print("4. 종료")
        print("-"*60)
        
        choice = input("선택하세요 (1-4): ").strip()
        
        if choice == "1":
            add_trade(calculator)
        elif choice == "2":
            calculator.print_summary()
        elif choice == "3":
            filename = input("파일명을 입력하세요 (엔터 시 자동 생성): ").strip()
            if not filename:
                filename = None
            calculator.save_to_excel(filename)
        elif choice == "4":
            print("프로그램을 종료합니다.")
            break
        else:
            print("잘못된 선택입니다. 1-4 중에서 선택하세요.")

def add_trade(calculator):
    """거래 정보 입력 및 기록"""
    print("\n거래 정보를 입력하세요:")
    
    try:
        trade_id = input("거래 ID: ").strip()
        if not trade_id:
            trade_id = len(calculator.trades) + 1
        
        entry_price = float(input("진입 가격: "))
        take_profit_pct = float(input("익절가 비율 (%): "))
        stop_loss_pct = float(input("스탑로스 비율 (%): "))
        
        # 익절/손절 여부 입력
        print("\n익절/손절 여부:")
        print("1. 익절")
        print("2. 손절")
        print("3. 청산가 입력 (자동 판단)")
        exit_choice = input("선택하세요 (1-3): ").strip()
        
        exit_price = None
        is_profit = None
        
        if exit_choice == "1":
            is_profit = True
        elif exit_choice == "2":
            is_profit = False
        elif exit_choice == "3":
            exit_price = float(input("청산 가격: "))
        else:
            print("잘못된 선택입니다. 익절로 처리합니다.")
            is_profit = True
        
        notes = input("메모 (선택사항): ").strip()
        
        # 투자 규칙 미리보기
        params = calculator.calculate_investment_params(stop_loss_pct)
        take_profit_price = entry_price * (1 + take_profit_pct / 100)
        stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
        
        print(f"\n[투자 규칙 미리보기]")
        print(f"진입가: {entry_price:,.2f}원")
        print(f"-" * 50)
        print(f"익절가 비율: {take_profit_pct:.2f}%")
        print(f"익절가: {take_profit_price:,.2f}원 (진입가 기준 +{take_profit_pct:.2f}%)")
        print(f"스탑로스 비율: {stop_loss_pct:.2f}%")
        print(f"스탑로스가: {stop_loss_price:,.2f}원 (진입가 기준 -{stop_loss_pct:.2f}%)")
        print(f"-" * 50)
        print(f"투입 금액: {params['investment_amount']:,.2f}원")
        print(f"레버리지: {params['leverage']}배")
        print(f"포지션 크기: {params['position_size']:,.2f}원")
        
        # 예상 수익/손실 계산
        if is_profit is True:
            expected_profit = params['investment_amount'] * params['leverage'] * (take_profit_pct / 100)
            print(f"예상 수익금: {expected_profit:,.2f}원")
        elif is_profit is False:
            expected_loss = params['investment_amount'] * params['leverage'] * (stop_loss_pct / 100)
            print(f"예상 손실금: {expected_loss:,.2f}원")
        
        confirm = input("\n이 정보로 거래를 기록하시겠습니까? (y/n): ").strip().lower()
        
        if confirm == 'y':
            trade = calculator.record_trade(
                trade_id=trade_id,
                entry_price=entry_price,
                take_profit_pct=take_profit_pct,
                stop_loss_pct=stop_loss_pct,
                exit_price=exit_price,
                is_profit=is_profit,
                notes=notes
            )
            
            print("\n[거래 기록 완료]")
            print(f"익절/손절: {trade['익절/손절']}")
            print(f"익절가: {trade['익절가격']:,.2f}원")
            print(f"스탑로스가: {trade['스탑로스가격']:,.2f}원")
            print(f"청산가: {trade['청산가격']:,.2f}원")
            print(f"수익/손실: {trade['수익금'] - trade['손실금']:,.2f}원")
            if trade['수익금'] > 0:
                print(f"  - 수익금: {trade['수익금']:,.2f}원")
                print(f"  - 재투자: {trade['재투자금액']:,.2f}원")
                print(f"  - 출금: {trade['출금금액']:,.2f}원")
            else:
                print(f"  - 손실금: {trade['손실금']:,.2f}원")
            print(f"총 투자 자산: {trade['총투자자산']:,.2f}원")
            print(f"총 자산: {trade['총자산']:,.2f}원")
        else:
            print("거래 기록이 취소되었습니다.")
            
    except ValueError as e:
        print(f"입력 오류: 숫자를 올바르게 입력하세요. ({e})")
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n프로그램이 중단되었습니다.")
        sys.exit(0)


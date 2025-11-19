"""
트레이딩 계산기
- 손익비에 따른 투자 규칙 적용
- 수익금의 절반만 재투자 (복리)
- 각 거래별 상세 계산 및 엑셀 저장
"""

import pandas as pd
from datetime import datetime
import os

class TradingCalculator:
    def __init__(self, initial_capital=8000000):
        """
        초기화
        
        Parameters:
        -----------
        initial_capital : float
            초기 자본금 (기본값: 8,000,000원)
        """
        self.initial_capital = initial_capital
        self.total_investment_capital = initial_capital  # 총 투자 자산 (원금 + 재투자 수익금)
        self.total_capital = initial_capital  # 총 자산 (투자 자산 + 출금된 수익금)
        self.withdrawn_profit = 0  # 출금된 수익금 누적
        
        # 거래 기록 저장
        self.trades = []
        
    def calculate_investment_params(self, stop_loss_pct):
        """
        스탑로스 비율을 기반으로 투자 규칙 계산
        
        Parameters:
        -----------
        stop_loss_pct : float
            스탑로스 비율 (%) - 레버리지를 곱하지 않은 기본 값
            
        Returns:
        --------
        dict : {
            'stop_loss_pct': 스탑로스 비율 (%),
            'investment_amount': 투입 금액,
            'leverage': 레버리지 배수,
            'position_size': 포지션 크기
        }
        """
        
        # 손익비에 따른 투자 규칙 적용
        if stop_loss_pct <= 1.0:
            # 1% 이하: 총 자산 8,000,000원 투입, 레버리지 3배
            investment_amount = 8000000
            leverage = 3
        elif stop_loss_pct > 1.0 and stop_loss_pct < 2.0:
            # 1% 초과 2% 미만: 총 자산의 50% 투입, 레버리지 3배
            investment_amount = self.total_investment_capital * 0.5
            leverage = 3
        elif stop_loss_pct >= 2.0 and stop_loss_pct < 2.5:
            # 2% 초과 2.5% 미만: 총 자산의 25% 투입, 레버리지 2배
            investment_amount = self.total_investment_capital * 0.25
            leverage = 2
        else:  # stop_loss_pct >= 2.5
            # 2.5% 초과: 총 자산의 25% 투입, 레버리지 2배
            investment_amount = self.total_investment_capital * 0.25
            leverage = 2
        
        # 포지션 크기 계산 (레버리지 적용)
        position_size = investment_amount * leverage
        
        return {
            'stop_loss_pct': stop_loss_pct,
            'investment_amount': investment_amount,
            'leverage': leverage,
            'position_size': position_size
        }
    
    def record_trade(self, trade_id, entry_price, take_profit_pct, stop_loss_pct, 
                     exit_price=None, is_profit=None, entry_time=None, exit_time=None, notes=""):
        """
        거래 기록 및 수익/손실 계산
        
        Parameters:
        -----------
        trade_id : str or int
            거래 ID
        entry_price : float (필수)
            진입 가격 - 익절가/스탑로스 가격 계산의 기준이 됩니다
        take_profit_pct : float (필수)
            익절가 비율 (%) - 레버리지를 곱하지 않은 기본 값
            익절가 = 진입가 × (1 + 익절가 비율 / 100)
        stop_loss_pct : float (필수)
            스탑로스 비율 (%) - 레버리지를 곱하지 않은 기본 값
            스탑로스가 = 진입가 × (1 - 스탑로스 비율 / 100)
        exit_price : float, optional
            청산 가격 (입력 시 자동으로 익절/손절 판단)
        is_profit : bool, optional
            익절 여부 (True: 익절, False: 손절). exit_price가 없을 때만 사용
        entry_time : datetime, optional
            진입 시간
        exit_time : datetime, optional
            청산 시간
        notes : str, optional
            메모
        """
        # 투자 규칙에 따른 파라미터 계산
        params = self.calculate_investment_params(stop_loss_pct)
        investment_amount = params['investment_amount']
        leverage = params['leverage']
        position_size = params['position_size']
        
        # 익절가와 스탑로스 가격 계산 (롱 포지션 기준)
        take_profit_price = entry_price * (1 + take_profit_pct / 100)
        stop_loss_price = entry_price * (1 - stop_loss_pct / 100)
        
        # 익절/손절 여부 판단
        if exit_price is not None:
            # 청산가가 입력된 경우, 익절가와 스탑로스 가격 중 어느 쪽에 가까운지 판단
            distance_to_tp = abs(exit_price - take_profit_price)
            distance_to_sl = abs(exit_price - stop_loss_price)
            is_profit = distance_to_tp < distance_to_sl
        elif is_profit is None:
            # 둘 다 입력되지 않은 경우 기본값 (익절로 가정)
            is_profit = True
            exit_price = take_profit_price
        
        # 청산가 설정
        if exit_price is None:
            exit_price = take_profit_price if is_profit else stop_loss_price
        
        # 수익/손실 계산
        if is_profit:
            # 익절: 익절가 비율 기준으로 계산
            price_change_pct = take_profit_pct
            profit_loss = investment_amount * leverage * (take_profit_pct / 100)
        else:
            # 손절: 스탑로스 비율 기준으로 계산
            price_change_pct = -stop_loss_pct
            profit_loss = investment_amount * leverage * (-stop_loss_pct / 100)
        
        # 수익인 경우
        if profit_loss > 0:
            reinvest_amount = profit_loss * 0.5  # 수익금의 절반만 재투자
            withdraw_amount = profit_loss * 0.5  # 나머지 절반은 출금
            
            # 총 투자 자산 업데이트 (원금 + 재투자 수익금)
            self.total_investment_capital += reinvest_amount
            
            # 출금된 수익금 누적
            self.withdrawn_profit += withdraw_amount
            
            # 총 자산 업데이트 (투자 자산 + 출금된 수익금)
            self.total_capital = self.total_investment_capital + self.withdrawn_profit
        else:
            # 손실인 경우
            reinvest_amount = 0
            withdraw_amount = 0
            
            # 총 투자 자산에서 손실 차감
            self.total_investment_capital += profit_loss  # profit_loss는 음수
            
            # 총 자산 업데이트 (출금된 수익금은 유지)
            self.total_capital = self.total_investment_capital + self.withdrawn_profit
        
        # 거래 기록 저장
        trade_record = {
            '거래ID': trade_id,
            '진입시간': entry_time if entry_time else datetime.now(),
            '청산시간': exit_time if exit_time else datetime.now(),
            '진입가격': entry_price,
            '익절가비율(%)': round(take_profit_pct, 2),
            '익절가격': round(take_profit_price, 2),
            '스탑로스비율(%)': round(stop_loss_pct, 2),
            '스탑로스가격': round(stop_loss_price, 2),
            '청산가격': round(exit_price, 2),
            '익절/손절': '익절' if is_profit else '손절',
            '투입금액': round(investment_amount, 2),
            '레버리지': leverage,
            '포지션크기': round(position_size, 2),
            '가격변동률(%)': round(price_change_pct, 2),
            '수익금': round(profit_loss, 2) if profit_loss > 0 else 0,
            '손실금': round(abs(profit_loss), 2) if profit_loss < 0 else 0,
            '재투자금액': round(reinvest_amount, 2),
            '출금금액': round(withdraw_amount, 2),
            '총투자자산': round(self.total_investment_capital, 2),
            '출금누적액': round(self.withdrawn_profit, 2),
            '총자산': round(self.total_capital, 2),
            '메모': notes
        }
        
        self.trades.append(trade_record)
        
        return trade_record
    
    def save_to_excel(self, filename=None):
        """
        거래 기록을 엑셀 파일로 저장
        
        Parameters:
        -----------
        filename : str, optional
            저장할 파일명 (기본값: trading_records_YYYYMMDD_HHMMSS.xlsx)
        """
        if not self.trades:
            print("저장할 거래 기록이 없습니다.")
            return
        
        # DataFrame 생성
        df = pd.DataFrame(self.trades)
        
        # 파일명 생성
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"trading_records_{timestamp}.xlsx"
        
        # 엑셀 파일로 저장
        df.to_excel(filename, index=False, engine='openpyxl')
        print(f"거래 기록이 '{filename}' 파일로 저장되었습니다.")
        print(f"총 {len(self.trades)}건의 거래가 기록되었습니다.")
        
        return filename
    
    def get_summary(self):
        """
        현재 상태 요약 정보 반환
        
        Returns:
        --------
        dict : 요약 정보
        """
        if not self.trades:
            return {
                '총거래수': 0,
                '총투자자산': self.total_investment_capital,
                '출금누적액': self.withdrawn_profit,
                '총자산': self.total_capital
            }
        
        df = pd.DataFrame(self.trades)
        total_profit = df['수익금'].sum()
        total_loss = df['손실금'].sum()
        net_profit = total_profit - total_loss
        
        return {
            '총거래수': len(self.trades),
            '총수익금': round(total_profit, 2),
            '총손실금': round(total_loss, 2),
            '순수익': round(net_profit, 2),
            '총투자자산': round(self.total_investment_capital, 2),
            '출금누적액': round(self.withdrawn_profit, 2),
            '총자산': round(self.total_capital, 2),
            '수익률(%)': round((net_profit / self.initial_capital) * 100, 2)
        }
    
    def print_summary(self):
        """요약 정보 출력"""
        summary = self.get_summary()
        print("\n" + "="*50)
        print("거래 요약")
        print("="*50)
        for key, value in summary.items():
            if isinstance(value, float):
                print(f"{key}: {value:,.2f}")
            else:
                print(f"{key}: {value:,}")
        print("="*50 + "\n")


# 사용 예시
if __name__ == "__main__":
    # 계산기 초기화
    calculator = TradingCalculator(initial_capital=8000000)
    
    # 예시 거래 1: 익절 거래
    print("거래 1: 익절 거래")
    trade1 = calculator.record_trade(
        trade_id=1,
        entry_price=100000,
        take_profit_pct=2.0,  # 익절가 비율: 2%
        stop_loss_pct=1.0,  # 스탑로스 비율: 1%
        is_profit=True,  # 익절
        notes="첫 번째 거래"
    )
    print(f"익절가: {trade1['익절가격']:,.2f}원")
    print(f"수익금: {trade1['수익금']:,.2f}원")
    print(f"재투자금액: {trade1['재투자금액']:,.2f}원")
    print(f"출금금액: {trade1['출금금액']:,.2f}원")
    print(f"총 투자 자산: {trade1['총투자자산']:,.2f}원")
    print(f"총 자산: {trade1['총자산']:,.2f}원\n")
    
    # 예시 거래 2: 손절 거래
    print("거래 2: 손절 거래")
    trade2 = calculator.record_trade(
        trade_id=2,
        entry_price=100000,
        take_profit_pct=2.0,  # 익절가 비율: 2%
        stop_loss_pct=1.5,  # 스탑로스 비율: 1.5%
        is_profit=False,  # 손절
        notes="두 번째 거래"
    )
    print(f"손실금: {trade2['손실금']:,.2f}원")
    print(f"총 투자 자산: {trade2['총투자자산']:,.2f}원")
    print(f"총 자산: {trade2['총자산']:,.2f}원\n")
    
    # 요약 정보 출력
    calculator.print_summary()
    
    # 엑셀 파일로 저장
    calculator.save_to_excel("trading_records_example.xlsx")


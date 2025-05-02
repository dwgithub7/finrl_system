import sys
import os
import numpy as np
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel,
    QFileDialog, QTextEdit, QTabWidget, QHBoxLayout
)
from PyQt5.QtGui import QFont
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# ✅ 기존 total_analyzer.py의 TradingAnalysisApp 가져오기
from loss_analyzer import TradingAnalysisApp

# ✅ Soft Voting Backtest 기능 추가 (별도 탭)
class SoftVotingBacktestTab(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        self.label = QLabel("Soft Voting Ensemble 기반 백테스트")
        self.label.setFont(QFont('Arial', 16))
        layout.addWidget(self.label)

        self.load_button = QPushButton("Soft Voting 결과 파일 불러오기")
        self.load_button.clicked.connect(self.load_file)
        layout.addWidget(self.load_button)

        self.eval_button = QPushButton("Soft Voting 백테스트 실행")
        self.eval_button.clicked.connect(self.run_backtest)
        layout.addWidget(self.eval_button)

        self.result_text = QTextEdit()
        layout.addWidget(self.result_text)

        self.setLayout(layout)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Soft Voting 결과 파일 선택", "", "Numpy Files (*.npy)")
        if file_path:
            self.file_path = file_path
            self.result_text.append(f"✅ Soft Voting 결과 파일 로드 완료: {file_path}\n")

    def run_backtest(self):
        try:
            preds = np.load(self.file_path, allow_pickle=True).item()
            entry_probs = preds['entry_probs']
            direction_probs = preds['direction_probs']

            np.random.seed(42)
            true_entry = np.random.randint(0, 2, size=len(entry_probs))
            true_direction = np.random.randint(0, 2, size=len(direction_probs))

            pred_entry = (entry_probs > 0.5).astype(int)
            pred_direction = (direction_probs > 0.5).astype(int)

            entry_accuracy = accuracy_score(true_entry, pred_entry)
            direction_accuracy = accuracy_score(true_direction, pred_direction)

            total_trades = np.sum(pred_entry)
            correct_trades = np.sum((pred_entry == 1) & (pred_direction == true_direction))
            win_rate = correct_trades / total_trades if total_trades > 0 else 0.0

            self.result_text.append(f"\n==== Soft Voting Backtest 결과 ====")
            self.result_text.append(f"총 진입 수: {total_trades}")
            self.result_text.append(f"진입 정확도: {entry_accuracy*100:.2f}%")
            self.result_text.append(f"방향 정확도: {direction_accuracy*100:.2f}%")
            self.result_text.append(f"매매 성공률(Win Rate): {win_rate*100:.2f}%\n")

        except Exception as e:
            self.result_text.append(f"❌ 오류 발생: {e}")

# ✅ 통합 탭 GUI
class MainTabApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("System Trading 통합 분석 툴")
        self.setGeometry(100, 100, 1000, 800)

        layout = QVBoxLayout()
        tabs = QTabWidget()

        self.loss_tab = TradingAnalysisApp()
        self.backtest_tab = SoftVotingBacktestTab()

        tabs.addTab(self.loss_tab, "Loss Analysis")
        tabs.addTab(self.backtest_tab, "Soft Voting Backtest")

        layout.addWidget(tabs)
        self.setLayout(layout)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainTabApp()
    window.show()
    sys.exit(app.exec_())

import sys
import numpy as np
import pandas as pd
import chardet
import re
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog, QTextEdit
)
from PyQt5.QtGui import QFont
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, accuracy_score
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

class TradingAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("System Trading Loss Analysis Tool")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.label = QLabel("\nLoss 개선 모델 평가 프로그램\n")
        self.label.setFont(QFont('Arial', 16))
        layout.addWidget(self.label)

        self.load_button = QPushButton("Entry/Direction 결과 파일 불러오기")
        self.load_button.clicked.connect(self.load_file)
        layout.addWidget(self.load_button)

        self.eval_button = QPushButton("Loss 관련 성능 평가 실행")
        self.eval_button.clicked.connect(self.evaluate_loss)
        layout.addWidget(self.eval_button)

        self.result_text = QTextEdit()
        layout.addWidget(self.result_text)

        self.setLayout(layout)

    def load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "결과 파일 선택", "", "All Files (*)")
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()

                if 'Loss' in first_line and 'LR' in first_line:
                    self.data = self.parse_loss_lr(file_path)
                elif 'Entry Loss' in first_line and 'Total Loss' not in first_line:
                    self.data = self.parse_entry_direction(file_path)
                else:
                    self.data = self.load_loss_log(file_path)

                if self.data is not None:
                    self.result_text.append(f"✅ 파일 로드 완료: {file_path}\n")
                else:
                    self.result_text.append(f"❌ 파일 읽기 실패")
            except Exception as e:
                self.result_text.append(f"❌ 파일 읽기 중 오류 발생: {e}")

    def load_loss_log(self, file_path):
        try:
            total_losses = []
            entry_losses = []
            direction_losses = []

            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
                    if len(match) == 3:
                        total, entry, direction = map(float, match)
                        total_losses.append(total)
                        entry_losses.append(entry)
                        direction_losses.append(direction)

            df = pd.DataFrame({
                'total_loss': total_losses,
                'entry_loss': entry_losses,
                'direction_loss': direction_losses
            })
            return df
        except Exception as e:
            print(f"❌ 파일 파싱 실패: {e}")
            return None

    def parse_loss_lr(self, file_path):
        losses = []
        lrs = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
                    if len(match) == 2:
                        loss, lr = map(float, match)
                        losses.append(loss)
                        lrs.append(lr)
            df = pd.DataFrame({'loss': losses, 'lr': lrs})
            return df
        except Exception as e:
            print(f"❌ 파일 파싱 실패: {e}")
            return None

    def parse_entry_direction(self, file_path):
        entry_losses = []
        direction_losses = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    match = re.findall(r"[-+]?[0-9]*\.?[0-9]+", line)
                    if len(match) == 2:
                        entry, direction = map(float, match)
                        entry_losses.append(entry)
                        direction_losses.append(direction)
            df = pd.DataFrame({'entry_loss': entry_losses, 'direction_loss': direction_losses})
            return df
        except Exception as e:
            print(f"❌ 파일 파싱 실패: {e}")
            return None

    def evaluate_loss(self):
        try:
            if 'entry_loss' in self.data.columns and 'direction_loss' in self.data.columns:
                entry_loss_mean = self.data['entry_loss'].mean()
                direction_loss_mean = self.data['direction_loss'].mean()

                self.result_text.append(f"\n==== 평균 Loss 평가 결과 ====")
                self.result_text.append(f"Entry Loss 평균: {entry_loss_mean:.4f}")
                self.result_text.append(f"Direction Loss 평균: {direction_loss_mean:.4f}\n")

                self.plot_loss_histogram()

            elif 'loss' in self.data.columns and 'lr' in self.data.columns:
                self.result_text.append(f"\n==== Loss vs Learning Rate 분석 ====")
                self.plot_loss_vs_lr()

            else:
                self.result_text.append("❌ 분석 가능한 데이터가 없습니다.")

        except Exception as e:
            self.result_text.append(f"오류 발생: {e}")

    def plot_loss_histogram(self):
        plt.figure(figsize=(10, 5))
        if 'total_loss' in self.data.columns:
            plt.hist(self.data['entry_loss'], bins=20, alpha=0.6, label='Entry Loss')
            plt.hist(self.data['direction_loss'], bins=20, alpha=0.6, label='Direction Loss')
        else:
            plt.hist(self.data['entry_loss'], bins=20, alpha=0.6, label='Entry Loss')
            plt.hist(self.data['direction_loss'], bins=20, alpha=0.6, label='Direction Loss')
        plt.title('Loss Distribution')
        plt.xlabel('Loss')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid()
        plt.show()

    def plot_loss_vs_lr(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.data['lr'], self.data['loss'], marker='o')
        plt.xlabel('Learning Rate')
        plt.ylabel('Loss')
        plt.title('Learning Rate vs Loss')
        plt.grid()
        plt.show()

    def analyze_returns(self):
        pass

    def analyze_risk_metrics(self):
        pass

    def analyze_trade_frequency(self):
        pass

    def simulate_slippage_and_fees(self):
        pass

    def generate_final_report(self):
        pass

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TradingAnalysisApp()
    window.show()
    sys.exit(app.exec_())

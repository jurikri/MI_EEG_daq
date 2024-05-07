# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 16:06:40 2024

@author: PC
"""

#%%
if True:
    import subprocess
    import pkg_resources
    
    def install_package(package_name):
        # 패키지가 이미 설치되어 있는지 확인
        installed_packages = {pkg.key for pkg in pkg_resources.working_set}
        if package_name not in installed_packages:
            # 패키지가 설치되어 있지 않을 경우 설치
            subprocess.check_call(["python", "-m", "pip", "install", package_name])
            print(f"{package_name} has been installed.")
        else:
            # 패키지가 이미 설치되어 있음
            print(f"{package_name} is already installed.")
    
    # 패키지 이름 설정
    package_names = ['pyOpenBCI', 'matplotlib', 'numpy', 'xmltodict', 'pyserial', 'requests', 'pynput']
    for package_name in package_names:
        install_package(package_name)

#%%

# FPS 확인코드
if False:
    from pyOpenBCI import OpenBCICyton
    import time
    import threading
    
    class SimpleEEGCounter:
        def __init__(self):
            self.callback_count = 0  # callback 함수 호출 횟수를 저장할 속성
            self.running = True  # 스트림이 실행 중인지 상태를 표시하는 플래그
    
        def callback(self, sample):
            self.callback_count += 1  # callback 함수가 호출될 때마다 카운트 증가
    
        def print_fps(self):
            while self.running:
                # 현재 카운트를 저장하고 리셋
                current_count = self.callback_count
                self.callback_count = 0
                print(f"FPS: {current_count}")
                time.sleep(1)  # 1초 대기
    
        def start(self, duration=60):
            # FPS 출력을 위한 스레드 시작
            fps_thread = threading.Thread(target=self.print_fps)
            fps_thread.start()
    
            # OpenBCICyton 객체 초기화 및 데이터 스트림 시작
            self.board = OpenBCICyton(port='COM4', daisy=False)
            self.board.start_stream(self.callback)
    
            # 지정된 시간(초) 동안 실행
            time.sleep(duration)
    
            # 스트림 중지 및 스레드 종료
            self.running = False
            self.board.stop_stream()
            fps_thread.join()  # FPS 스레드가 종료될 때까지 대기
    
            print("Data collection complete.")
    
    if __name__ == "__main__":
        eeg_counter = SimpleEEGCounter()
        eeg_counter.start()  # 기본적으로 60초 동안 실행

#%%

import numpy as np
import matplotlib.pyplot as plt
from pyOpenBCI import OpenBCICyton
import pickle
from datetime import datetime
import sys
import multiprocessing
import time
import queue
import os
from pynput import keyboard, mouse
import logging
import time
from datetime import datetime

current_path = os.getcwd()

plt.ion()

class MultiChannelEEGPlot:
    def __init__(self, queue, channels=[0, 1, 2], num_samples=2500, update_interval=25):
        self.queue = queue
        self.channels = channels
        self.num_samples = num_samples
        self.update_interval = update_interval
        self.data = {channel: np.zeros(self.num_samples) for channel in channels}
        self.fig, self.axs = plt.subplots(len(channels), 1, figsize=(10, 7))
        self.lines = {channel: ax.plot([], [], 'r-')[0] for channel, ax in zip(channels, self.axs)}
        self.start_time = time.time()  # 그래프 업데이트 시작 시간 기록
        self.queue = queue
        self.is_running = True  # 여기에서 is_running 속성을 정의
        self.update_counter = 0  # 이 카운터로 업데이트 간격을 조절
        self.downsample_factor = 2  # 다운샘플링을 위한 인자

        for ax in self.axs:
            ax.set_xlim(0, self.num_samples // self.downsample_factor)
            ax.set_ylim(-8000, 8000)

    def update_plot(self):
        sample_rate = 250  # 샘플레이트, 예를 들어 250Hz
        window_size = 5 * sample_rate  # 5초간의 데이터 수, 예: 5 * 250 = 1250
        update_y_axis_interval = 5  # y축 업데이트 간격, 초 단위
        # smoothing_window = 50  # 스무딩을 위한 데이터 윈도우 크기
        last_update_time = time.time()

        while self.is_running:
            current_time = time.time()
            try:
                data = self.queue.get_nowait()  # 큐에서 데이터 가져오기
                # 데이터를 내부 버퍼에 추가하는 코드
                for channel in self.channels:
                    self.data[channel] = np.roll(self.data[channel], -1)
                    self.data[channel][-1] = data[channel]

                self.update_counter += 1
                
                if self.update_counter >= self.update_interval:
                    for channel in self.channels:
                        downsampled_data = self.data[channel][::self.downsample_factor]
                        self.lines[channel].set_data(np.arange(len(downsampled_data)), downsampled_data)
                    
                    # 매 5초마다 y축 업데이트
                    if current_time - last_update_time >= update_y_axis_interval:
                        for channel in self.channels:
                            # 최근 5초간의 데이터 선택
                            recent_data = self.data[channel][-window_size:]
                            mean = np.mean(recent_data)
                            std = np.std(recent_data)
                            lower_bound = mean - 3*std
                            upper_bound = mean + 3*std
                            self.axs[channel].set_ylim(lower_bound, upper_bound)
                        
                        last_update_time = current_time

                    self.fig.canvas.draw()
                    self.fig.canvas.flush_events()
                    self.update_counter = 0  # 카운터 리셋

            except queue.Empty:
                time.sleep(0.01)  # 큐가 비어 있으면 잠시 대기
                continue
            
    def stop(self):
        self.is_running = False

def data_collection(queue):
    callback_count = 0  # callback 함수 호출 횟수를 저장할 변수
    full_data = []  # 모든 데이터를 누적할 리스트
    start_time = None  # 첫 callback 호출 시간
    save_interval = 5  # 데이터 저장 간격 (초)
    last_save_time = time.time()  # 마지막 저장 시간

    def callback(sample):
        nonlocal callback_count, start_time, last_save_time
        if start_time is None:
            start_time = time.time()  # 첫 callback 시간 기록

        callback_count += 1

        # FPS 계산 및 출력
        current_time = time.time()
        if current_time - start_time >= 1.0:
            print(f"FPS: {callback_count}")
            callback_count = 0
            start_time = current_time

        # 샘플 데이터를 리스트로 변환하여 누적
        current_time = time.time()
        data = [sample.channels_data[channel] for channel in range(8)] + [current_time] 
        full_data.append(data)  # 누적 데이터에 추가
        queue.put(data)  # GUI 업데이트를 위해 큐에 데이터 추가
        
        if current_time - last_save_time >= save_interval:
            filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S.pkl")
            filename = os.path.join(current_path, 'data', filename)
            with open(filename, 'wb') as file:
                pickle.dump(full_data, file)
                print(f"Data saved to {filename}")
                full_data.clear()  # 저장 후 데이터 클리어
            last_save_time = current_time

    board = OpenBCICyton(port='COM4', daisy=False)
    board.start_stream(callback)

    # 데이터 수집은 사용자가 종료할 때까지 계속 실행
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        board.stop_stream()  # 사용자 인터럽트에 의해 스트림 중지
        print("Data collection finished. Exiting program.")

if __name__ == "__main__":
    data_queue = multiprocessing.Queue()
    data_process = multiprocessing.Process(target=data_collection, args=(data_queue,))

    data_process.start()

    plot = MultiChannelEEGPlot(data_queue)
    try:
        plot.update_plot()
    finally:
        plot.stop()
        data_process.join()
        
    # 마지막 기록 시간 및 빈도 제한

    


#%%



from pyOpenBCI import OpenBCICyton

def stream_data():
    # List to hold the EEG data and timestamps
    eeg_data = []

    # Callback function to append data samples and timestamps
    def append_sample(sample):
        # Append the channel data and timestamp to the eeg_data list
        eeg_data.append({
            'channels_data': sample.channels_data,
            'timestamp': sample.timestamp
        })
        # Optional: Print the data and timestamp to see it in real-time
        print('Timestamp:', sample.timestamp, 'Channels Data:', sample.channels_data)

    # Connect to the OpenBCI Cyton board
    board = OpenBCICyton(port='COM4', daisy=False)
    
    # Start streaming data, passing append_sample as the callback
    print("Starting data stream...")
    board.start_stream(append_sample)

    # After streaming, return the collected data
    return eeg_data

if __name__ == '__main__':
    try:
        data = stream_data()
        print("Data collection complete.")
        # Now `data` holds all your EEG samples and timestamps, and you can process it as needed.
        print("Number of samples collected:", len(data))
    except KeyboardInterrupt:
        print("Stream stopped by user.")
    except Exception as e:
        print(f"An error occurred: {e}")








































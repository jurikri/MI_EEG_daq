# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 13:19:23 2024

@author: PC
"""


#%%
time_points=250; channels=2; sampling_rate=250
def msmain(time_points=250, channels=2, sampling_rate=250):
    #%%
    import numpy as np
    import tensorflow as tf
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, Layer, Concatenate, Dense, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.initializers import Initializer
    # from tensorflow.keras.layers import GlobalAveragePooling1D
    from tensorflow.keras.layers import Lambda
    # import tensorflow as tf
    from tensorflow.keras.constraints import NonNeg

    
    # Morlet 웨이블릿의 실수 부분을 계산하는 함수
    def morlet_real(sampling_rate, f0, kernel_size):
        t_size = kernel_size / sampling_rate
        t = np.arange(-t_size / 2, t_size / 2, 1 / sampling_rate)
        s = 1 / (2 * np.pi * f0)
        wavelet = np.pi**(-0.25) * np.exp(1j * 2 * np.pi * f0 * t) * np.exp(-t**2 / (2 * s**2))
        return np.real(wavelet)[:kernel_size]

    class MorletInitializer(Initializer):
        def __init__(self, sampling_rate, f0, kernel_size):
            super(MorletInitializer, self).__init__()
            self.weights = morlet_real(sampling_rate, f0, kernel_size)
    
        def __call__(self, shape, dtype=None):
            input_channels = 1
            weights_reshaped = np.reshape(self.weights, (len(self.weights), input_channels, 1))
            return weights_reshaped.astype(np.float32)
        
    # Create a custom layer to combine channels
    class CombineChannels(Layer):
        def __init__(self, beta1, beta2, **kwargs):
            super(CombineChannels, self).__init__(**kwargs)
            self.beta1 = beta1
            self.beta2 = beta2
        
        def call(self, inputs):
            ch1, ch2 = inputs
            combined = ch1 * self.beta1 - ch2 * self.beta2
            return tf.expand_dims(combined, -1)
        
        # 20개의 beta 조합 생성
    beta_combinations = [(tf.Variable(1, trainable=True, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, 10)),
                          tf.Variable(1, trainable=True, dtype=tf.float32, constraint=lambda x: tf.clip_by_value(x, 0, 10)))
                         for _ in range(2)]
    
    input_layer = Input(shape=(time_points, channels)) # 모델의 입력 레이어 정의
    fi_outputs = []
    for fi in range(2, 51):
        # 각 fi에 대한 처리 결과를 저장할 리스트
        fi_filter_outputs = []
        
        for beta1, beta2 in beta_combinations:
            # CombineChannels 인스턴스 생성
            combined_signal_layer = CombineChannels(beta1, beta2)
            combined_signal = combined_signal_layer([input_layer[:, :, 0], input_layer[:, :, 1]])
      
            kernel_size = sampling_rate // fi
            morlet_init = MorletInitializer(sampling_rate, fi, kernel_size)
            conv_layer = Conv1D(
                filters=1, 
                kernel_size=kernel_size, 
                strides=max(1, kernel_size // 10),
                activation='relu', 
                kernel_initializer=morlet_init,
                kernel_constraint=NonNeg(),
                bias_constraint=NonNeg(),
                trainable=False
            )(combined_signal)
            
            flattened_output = Flatten()(conv_layer)
            max_abs_output = Lambda(lambda x: tf.reduce_mean(x, axis=1, keepdims=True))(flattened_output)
            fi_filter_outputs.append(max_abs_output)
    
        # 각 fi에 대한 모든 beta 조합의 결과를 병합
        # concatenated_fi = Concatenate()(fi_filter_outputs)
        # fi_outputs.append(concatenated_fi)
        fi_combined_output = Concatenate()(fi_filter_outputs)
        fi_single_output = Dense(1, activation='linear')(fi_combined_output)
        fi_outputs.append(fi_single_output)
    
    # 모든 fi에 대해 병합된 결과를 최종적으로 연결
    concatenated_fi_output = Concatenate(name='concatenated_fi_output')(fi_outputs)
    classification_output = Dense(2, activation='softmax')(concatenated_fi_output)
    final_model = Model(inputs=input_layer, outputs=classification_output)
    final_model.compile(optimizer=Adam(), loss='categorical_crossentropy')

    return final_model
#%%
if False:
    final_model = msmain(time_points=250, channels=2, sampling_rate=250)
    from tensorflow.keras.utils import plot_model
    
    # 'final_model'이라는 이름의 모델을 가정하고 있습니다
    with open(r'C:\SynologyDrive\worik in progress\SOSOdevice\VNS_PPG_1' + \
              '\\model_summary.txt', 'w') as f:
        final_model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    # 추가적으로, 모델을 시각화하고자 한다면 다음 코드를 사용하여 이미지 파일로 저장할 수 있습니다
    plot_model(final_model, to_file='model_structure.png', show_shapes=True)
    
    import tensorflow as tf
    # 모델의 각 Conv1D 레이어의 필터 크기를 출력
    for i, layer in enumerate(final_model.layers):
        # 레이어 타입이 Conv1D인 경우
        if isinstance(layer, tf.keras.layers.Conv1D):
            config = layer.get_config()
            # 필터(커널) 크기 출력
            print(f"Layer {i} ({layer.name}): Filter size = {config['kernel_size']}")
    


# # 커널 예시 들어서 한번보고
# # 전처리에서 스케일에 대한 처리가 필요없는지 고려해볼것
# # 마지막 fi에서 양 / 음수 가 유지되는지 생각해볼것









































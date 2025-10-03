import numpy as np
from gnuradio import gr
from scipy.signal import firwin2, lfilter

#granularity
N = 1000

class firwin2_filter(gr.sync_block):
    def __init__(self, length=91, sinc_exp=1.0, sinc_gain=110.0, sinc_ratio=0.125):
        gr.sync_block.__init__(self,
            name="Sinc Compensating Filter",
            in_sig=[np.complex64],
            out_sig=[np.complex64])
        self.length, self.sinc_exp, self.sinc_gain, self.sinc_ratio = length, sinc_exp, sinc_gain, sinc_ratio
        self.freq = np.linspace(0.0, 1.0, N)
        self.ampl = np.zeros(N)
        self.update_taps()
        self.zi_r = np.zeros(len(self.taps)-1, dtype=np.float32)
        self.zi_i = np.zeros(len(self.taps)-1, dtype=np.float32)   

    def update_taps(self):
        for i in range(N-1):
            self.ampl[i] = pow(1 + pow((1 - np.sinc(i/(N-1) * self.sinc_ratio)), self.sinc_exp), self.sinc_gain)
        self.ampl[N-1] = 0
        self.taps = firwin2(self.length, self.freq, self.ampl, window='blackmanharris').astype(np.float32)
        
    #the set_* functions below do not actually make the parameters runtime adjustable :(
    def set_length(self, length):
        self.length = length
        self.update_taps()
        self.zi_r = np.zeros(len(self.taps)-1, dtype=np.float32)
        self.zi_i = np.zeros(len(self.taps)-1, dtype=np.float32)

    def set_sinc_exp(self, sinc_exp):
        self.sinc_exp = sinc_exp
        self.update_taps()

    def set_sinc_gain(self, sinc_gain):
        self.sinc_gain = sinc_gain
        self.update_taps()

    def set_sinc_ratio(self, sinc_ratio):
        self.sinc_ratio = sinc_ratio
        self.update_taps()

    def work(self, input_items, output_items):
        x = input_items[0]
        y_r, self.zi_r = lfilter(self.taps, 1.0, x.real, zi=self.zi_r)
        y_i, self.zi_i = lfilter(self.taps, 1.0, x.imag, zi=self.zi_i)
        output_items[0][:] = y_r + 1j * y_i
        return len(output_items[0])

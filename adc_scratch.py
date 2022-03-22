import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
from eo_testing.utils.array_tools import get_true_windows
import scipy.fftpack as sfp

min_expected = 200

def model_signal(sparse_yf, xf, phases, n_samples):
    n = len(sparse_yf)
    sparse_yf = sparse_yf[:n//2]
    xf = xf[:n//2]
    phases = phases[:n//2]
    nz = np.nonzero(sparse_yf)[0]
        
    #print(xf[4:12], phases[4:12])
    freqs = xf[nz]
    phis = phases[nz].reshape(1, len(freqs))
    coefs = 2*sparse_yf[nz].reshape(1, len(freqs))/n_samples
    freqs = freqs.reshape(1, len(freqs))
                            
    #print(freqs, phis/np.pi, np.abs(coefs))
    def model(x):
        x = x.reshape(1, len(x))
        result = np.abs(coefs) @ np.cos(2*np.pi*(freqs.transpose() @ x) - phis.transpose())
        return result.flatten()                                               

    return model

adcmax = 2**18

with open('analysis/adc/test_data/test_data.pkl', 'rb') as f:
    test_data = pkl.load(f)

x = np.arange(0, adcmax+1)

#fig = plt.figure(figsize=(30, 20))
fig, axs = plt.subplots(2, 1, figsize=(30, 20), sharex=True)
y_arrs = []
w_arrs = []

colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(test_data['observed']['C17'].keys())))

for iexp, (expnum, bindict) in enumerate(test_data['observed']['C17'].items()):
    c = colors[iexp]
    obs = np.array([bindict[b] if b in bindict.keys() else np.nan for b in x])
    edict = test_data['expected']['C17'][expnum]
    exp = np.array([edict[b] if b in edict.keys() else np.nan for b in x])
    #exp *= obs[np.isfinite(obs)].sum() / exp[np.isfinite(exp)].sum()
    cond = exp > min_expected
    y = np.nan*x
    y[cond] = obs[cond] / exp[cond] - 1
    w = 0*x
    w[cond] = exp[cond]/4 ##weighting will be slightly different in dnlTask.py
    w_arrs.append(w)
    y_arrs.append(y)
    #plt.plot(x, obs, color=c, label='Observed', ds='steps-post')
    #plt.plot(x, exp, label='Expected', ds='steps-post')
    xerr = 0.5 * np.ones(len(x))
    #axs[0].plot(x, obs, color=c, label='Observed', ds='steps-post')
    #axs[0].plot(x, exp, label='Expected', ds='steps-post')
    #axs[1].errorbar(x+.5, y, yerr=np.sqrt(exp)/exp, xerr=xerr, alpha=0.5, ls='None')

#plt.title('12781 R10 S00 C17 Distributions', fontsize=24)
#plt.xlabel('Signal', fontsize=18)
#plt.ylabel('Counts', fontsize=18)
#axs[0].grid(visible=True)
#axs[1].grid(visible=True)
#axs[0].set_xlim(58000, 58500)
#axs[0].set_ylim(400, 900)
#axs[1].set_xlim(58000, 58500)
#axs[0].set_title('12781 R10 S00 C17 Distributions', fontsize=24)
#axs[1].set_title('12781 R10 S00 C17 DNL Measurement', fontsize=24)
#axs[1].set_xlabel('Signal', fontsize=18)
#axs[0].set_ylabel('Counts', fontsize=18)
#axs[1].set_ylabel('DNL', fontsize=18)
#plt.legend()
#plt.show()
#fig.savefig('analysis/adc/plots/examples/12781_C17_dists_dnl.png')
#exit()

#finite = np.isfinite(y_arrs)
#i01 = np.logical_and(finite[0], finite[1])
#i02 = np.logical_and(finite[0], finite[2])
#i12 = np.logical_and(finite[1], finite[2])
#
#c01 = np.corrcoef(y_arrs[0][i01], y_arrs[1][i01])[0, 1]
#c02 = np.corrcoef(y_arrs[0][i02], y_arrs[2][i02])[0, 1]
#c12 = np.corrcoef(y_arrs[1][i12], y_arrs[2][i12])[0, 1]
#
#print(c01, c02, c12)

y_arrs = np.array(y_arrs)
w_arrs = np.array(w_arrs)
cond = np.isfinite(y_arrs)
sw = y_arrs * w_arrs
sw[~cond] = 0
w_arrs[~cond] = 0
avg = sw.sum(axis=0)/w_arrs.sum(axis=0)
print(len(avg[np.isfinite(avg)]), avg.itemsize)
exit()
windows = get_true_windows(np.isfinite(avg))
colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(windows)))

dnl_curve = np.nan * x

for iw, window in enumerate(windows):
    c = colors[iw]
    n_samples = len(window)

    if n_samples < 200:
        continue

    #for i in range(8, 18):
    #    if 2**i > n_samples:
    #        ncoef = 2**i
    #        break
    #    else:
    #        continue


    ##yf = sfp.fft(avg[window], n=ncoef)
    yf = sfp.fft(avg[window])
    nf = len(yf)
    ncoef = nf
    xf = sfp.fftfreq(nf, d=1)

    ##binary_frequencies = 1/np.array([2**p for p in range(int(np.log2(ncoef)))])
    ##binary_mode_cond = np.in1d(xf, binary_frequencies)

    ###print(.25 in xf)

    yf_pk = yf[:nf//2]
    abs_yf_pk = np.abs(yf_pk)
    ##yf_cutoff = np.max(abs_yf_pk)

    ###for j in range(1):#10):
    ###    yf_cutoff = np.max(abs_yf_pk[abs_yf_pk < yf_cutoff])

    newyf = np.zeros(len(yf), dtype=np.complex128)
    ##newyf[np.abs(yf)>=yf_cutoff] = yf[np.abs(yf)>=yf_cutoff]
    ##newyf[binary_mode_cond] = yf[binary_mode_cond]
    ##newyf = np.append(newyf, yf[-1])
    ##newxf = np.append(xf, 1/2)


    quant = 0.85
    dominant_mode_condition = np.abs(yf)>np.quantile(abs_yf_pk, quant)
    newyf[dominant_mode_condition] = yf[dominant_mode_condition]

    ##newyf[np.abs(yf)>yf_cutoff] = yf[np.abs(yf)>yf_cutoff]


    #plt.plot(xf[:ncoef//2], 0.03*iw + 2/n_samples * np.abs(yf[:ncoef//2]), color=c, label=iw)
    ##plt.plot(xf[:ncoef//2], 0.03*iw + 2/n_samples * np.abs(newyf[:ncoef//2]), color=c, label=iw)

    phases = -np.arctan2(newyf.imag, newyf.real)
    #plt.plot(xf[:ncoef//2], iw + phases[:ncoef//2]/np.pi, color=c, label=iw)

    plt.plot(x[window], avg[window], color=c, label=iw, ds='steps-post')

    model = model_signal(newyf, xf, phases, n_samples)
    current_window_min = x[window].min()

    try:
        next_x_window = x[windows[iw+1]] 
        if len(next_x_window) < 200:
            next_x_window = x[windows[iw+2]]
        next_window_min = next_x_window.min()
    except IndexError:
        dnl_curve[x[window]] = avg[window]
        continue

    #tx = np.arange(2*len(x[window]))
    tx = np.arange(next_window_min - current_window_min)
    modeled_signal = model(tx)
    rescaled_model = modeled_signal * avg[window].std() / modeled_signal.std()
    mx = np.arange(current_window_min, next_window_min)
    #mx = np.arange(current_window_min, current_window_min + 2 * len(x[window]))

    ##plt.plot(mx, modeled_signal, label='modeled+extended', ds='steps-post')
    #plt.plot(mx, rescaled_model, label='rescaled', ds='steps-post')
    #print(np.corrcoef(avg[window], rescaled_model[np.in1d(mx, x[window])])[0, 1])
    ##print(rescaled_model[0], rescaled_model[len(rescaled_model)//2])

    dnl_curve[mx] = rescaled_model
    dnl_curve[x[window]] = avg[window]
    #print(avg[window].mean())
    #break


plt.plot(x, dnl_curve, ds='steps-post')
plt.legend()
plt.xlabel('Signal', fontsize=18)
plt.ylabel('DNL', fontsize=18)
#plt.xlabel(r'Frequency (ADU$^{-1}$)', fontsize=18)
#plt.ylabel(r'$|C_k|$', fontsize=18)
#plt.ylabel(r'$\phi_k/\pi$ (Radians)', fontsize=18)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.title('DNL vs Flux', fontsize=24)
#plt.title('DNL FFT', fontsize=24)
#plt.title('DNL Mode Phase', fontsize=24)
#plt.xlim(125000, 125200)
#plt.xlim(30200, 30300)
plt.grid(visible=True)
plt.show()
fig.savefig('analysis/adc/plots/examples/12781_C17_DNL_inpainted.png')
#fig.savefig('analysis/adc/plots/examples/12781_C17_DNL_FFT.png')
#fig.savefig('analysis/adc/plots/examples/12781_C17_DNL_phases.png')


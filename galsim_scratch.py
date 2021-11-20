import galsim
import os
import numpy as np
import pickle as pkl
import copy
import matplotlib.ticker as mtick
import sys
from matplotlib import pyplot as plt
from collections import defaultdict
from eo_testing.adc.adcTask import adc_bit_dataset
from scipy import interpolate
np.set_printoptions(suppress=True, linewidth=np.nan, threshold=sys.maxsize)

class galsim_dnl_dataset():
    def __init__(self):#, ds_dict, n_shears):
        self.means = defaultdict(dict)
        self.sigmas = defaultdict(dict) 
        self.fluxes = defaultdict(dict) 
        self.g1s = defaultdict(dict) 
        self.g2s = defaultdict(dict) 
        self.centroid_xs = defaultdict(dict) 
        self.centroid_ys = defaultdict(dict) 

def save_dataset(dataset, filename):
    with open(filename, 'wb') as f:
        pkl.dump(dataset, f)
    print(f'Wrote {filename}')

def make_blank_image(flux, g1, g2):
    # random seed
    #rng = galsim.BaseDeviate(12345)
    
    # model atmospheric turbulence as a VonKarman, with some small shear
    gprof = galsim.VonKarman(lam=700., r0=0.2, L0=10.0)#.shear(g1=0.01, g2=0.00)
    
    # add 4.5 microns sigma of Gaussian to model diffusion
    # convert 4.5 microns to arcsec with factor 0.2"/10micron 
    pixscale = 0.2/10.e-6
    dprof = galsim.Gaussian(sigma=4.5e-6*pixscale)
    
    # add optical term, making up reasonable Zernike terms
    oprof = galsim.OpticalPSF(lam=700.0, diam=8.4,
                              defocus = 0.1, coma1 = -0.1, coma2 = 0.005, astig1 = 0.1, astig2 = 0.005,
                              obscuration = 0.4)
    
    # convolve these terms
    prof = galsim.Convolve([gprof, dprof, oprof]).withFlux(flux).shear(g1=g1, g2=g2)
    
    # draw image
    blank_image = galsim.Image(32, 32, scale=0.2, xmin=0, ymin=0, dtype=np.float64)  
    return blank_image, prof

def plot_star_image(im_dict, inshear, influx, im_num):
    im_arr = im_dict[inshear][influx][im_num]

    # DNL bins
    dnl_bins = make_dnl_bins()
    pedestal = 2**8

    star_image = galsim.Image(np.floor(im_arr), dtype=int, xmin=0, ymin=0)
    adc_image = make_adc_image(im_arr, pedestal, dnl_bins)
    f, axs = plt.subplots(1, 2, figsize=(22, 11))
    star_moments = star_image.FindAdaptiveMom(weight=None, strict=False)
    adc_moments = adc_image.FindAdaptiveMom(weight=None, strict=False)

    #print(np.floor(im_arr), adc_image.array)
    diff = star_image.array - adc_image.array
    idx = np.where(diff>100) 
    print(star_image.array[idx], adc_image.array[idx])
    print(star_moments.moments_amp)
    print(star_moments.moments_sigma)
    print(star_moments.observed_shape.g1, star_moments.observed_shape.g2)
    print(star_moments.moments_centroid.x, star_moments.moments_centroid.y)

    print(adc_moments.moments_amp)
    print(adc_moments.moments_sigma)
    print(adc_moments.observed_shape.g1, adc_moments.observed_shape.g2)
    print(adc_moments.moments_centroid.x, adc_moments.moments_centroid.y)
 
 
    im0 = axs[0].imshow(star_image.array, origin='lower', interpolation='None')
    im1 = axs[1].imshow(adc_image.array, origin='lower', interpolation='None')
    axs[0].set_title('Ideal')
    axs[1].set_title('DNL')
    cbar_ax = f.add_axes([0.93, 0.2, 0.025, 0.6])
    f.colorbar(im0, cax=cbar_ax)
    f.suptitle('Galaxies', fontsize=20)
    #f.savefig('analysis/adc/plots/galsim/galaxy.png')
    plt.show()

def read_star_images(filename):
    with open(filename, 'rb') as f:
        im_dict = pkl.load(f)
    return im_dict

def write_star_images(input_fluxes, input_shears, n_ims=500):
    im_dict = defaultdict(dict)
    write_dir = '/gpfs/slac/lsst/fs1/u/adriansh/analysis/adc/galsim/star_images'
    fname = 'star_images.pkl'
    filename = os.path.join(write_dir, fname)

    for inshear in input_shears:
        for influx in input_fluxes:
            im_dict[inshear][influx] = np.zeros((n_ims, 32, 32))
    for (g1, g2), fdict in im_dict.items():
        for iflux, im_arrs in fdict.items():
            print(g1, g2, iflux)
            for n in range(n_ims):
                star_image = make_star_image(iflux, g1, g2)
                im_arrs[n] = star_image.array

    with open(filename, 'wb') as f:
        pkl.dump(im_dict, f)
    print(f'Wrote {filename}')

def make_star_image(flux, g1, g2):
    #rng = galsim.BaseDeviate(12345)
    rng = galsim.BaseDeviate() # no seed
    blank_image, prof = make_blank_image(flux, g1, g2)
    star_image = prof.drawImage(image=blank_image, method='auto')
        
    # generate noise and add to image
    noise = galsim.CCDNoise(rng, gain=1.0, read_noise=5.0, sky_level=300.0)
    star_image.addNoise(noise)
    return star_image

def make_corrected_image(array, pedestal):
    array += pedestal
    for i, row in enumerate(array):
        for j, signal in enumerate(row):
            binary_signal = f'{signal:018b}'
            adc_signal = list(binary_signal)
            adc_signal[-1] = str(np.random.binomial(1, 0.5))
            adc_signal[-2] = str(np.random.binomial(1, 0.5))
            adc_signal = ''.join(adc_signal)
            array[i, j] = int(adc_signal, 2)

    corrected_image = galsim.Image(array - pedestal, dtype=int, xmin=0, ymin=0)
    return corrected_image

def make_adc_image(im_array, pedestal, dnl_bins):
    #x = np.arange(np.floor(dnl_bins.min()), np.ceil(dnl_bins.max()))
    im_arr = np.digitize(im_array+pedestal, dnl_bins) - 1

    #if bin_ind.min() < 0:
    #    raise ValueError("Digitized array bin index less than zero")

    #im_array = bin_ind + pedestal

    # subtract pedestal, make adc and diff images
    adc_image = galsim.Image(im_arr-pedestal, dtype=int, xmin=0, ymin=0)
    return adc_image
 
def make_dataset(im_dict):
    for infl, sd in im_dict.items():
        for insh, ims in sd.items():
            n_iter = len(ims)
            break
        break

    # Number of images (ideal, dnl)#, corrected)
    n_ims = 2
   
    # DNL bins
    dnl_bins = make_dnl_bins()
    pedestal = 2**8

    # Make a dataset
    dataset = galsim_dnl_dataset()
    ds_dict = dict(Ideal = np.zeros(n_iter),
                   ADC = np.zeros(n_iter))
                   #Corrected = np.zeros(n_iter))

    for ish in range(len(input_shears)):
        for iflux in input_fluxes:
            dataset.means[ish][iflux] = copy.deepcopy(ds_dict)
            dataset.sigmas[ish][iflux] = copy.deepcopy(ds_dict) 
            dataset.fluxes[ish][iflux] = copy.deepcopy(ds_dict) 
            dataset.g1s[ish][iflux] = copy.deepcopy(ds_dict) 
            dataset.g2s[ish][iflux] = copy.deepcopy(ds_dict) 
            dataset.centroid_xs[ish][iflux] = copy.deepcopy(ds_dict) 
            dataset.centroid_ys[ish][iflux] = copy.deepcopy(ds_dict) 

    for ishear, ((g1, g2), fdict) in enumerate(im_dict.items()):
        print(g1, g2)
        for input_flux, im_arrs in fdict.items():
            means = np.zeros((n_ims, n_iter)) #ideal, dnl, corrected
            sigmas = np.zeros((n_ims, n_iter))
            fluxes = np.zeros((n_ims, n_iter))
            g1s = np.zeros((n_ims, n_iter))
            g2s = np.zeros((n_ims, n_iter))
            cxs = np.zeros((n_ims, n_iter))
            cys = np.zeros((n_ims, n_iter))

            for n, arr in enumerate(im_arrs):
                star_image = galsim.Image(arr, xmin=0, ymin=0)

                # digitize and add pedestal
                int_arr = np.floor(arr)
                int_image = galsim.Image(int_arr, dtype=int, xmin=0, ymin=0)
                adc_image = make_adc_image(arr, pedestal, dnl_bins)
                #corrected_image = make_corrected_image(np.array(adc_image.array), pedestal)
                images = [int_image, adc_image]#, corrected_image]

                for i_image, image in enumerate(images):
                    # calculate HSM moments (these are in pixel coordinates)
                    moments = image.FindAdaptiveMom(weight=None, strict=False)
                    means[i_image, n] = np.mean(image.array)
                    sigmas[i_image, n] = moments.moments_sigma
                    fluxes[i_image, n] = moments.moments_amp
                    g1s[i_image, n] = moments.observed_shape.g1
                    g2s[i_image, n] = moments.observed_shape.g2
                    cxs[i_image, n] = moments.moments_centroid.x
                    cys[i_image, n] = moments.moments_centroid.y

            for ik, k in enumerate(ds_dict.keys()):
                dataset.means[ishear][input_flux][k] = means[ik]
                dataset.sigmas[ishear][input_flux][k] = sigmas[ik] 
                dataset.fluxes[ishear][input_flux][k] = fluxes[ik] 
                dataset.g1s[ishear][input_flux][k] = g1s[ik] 
                dataset.g2s[ishear][input_flux][k] = g2s[ik] 
                dataset.centroid_xs[ishear][input_flux][k] = cxs[ik] 
                dataset.centroid_ys[ishear][input_flux][k] = cys[ik] 

    write_to = 'analysis/adc/datasets/galsim'
    fname = 'galsim_dnl_dataset.pkl'
    filename = os.path.join(write_to, fname)
    save_dataset(dataset, filename)

def plot_vals(input_shears):
    filename = 'analysis/adc/datasets/galsim/galsim_dnl_dataset.pkl'
    save_to = 'analysis/adc/plots/galsim/'
    titles = ['Flux', 'Sigma', 'Shear', 'Centroid']
    ylabels = ['Flux', 'Sigma', [r'$\gamma_1$', r'$\gamma_2$'], ['$x_0$', r'$y_0$']]

    x_data = defaultdict(list)
    x_err = defaultdict(list)
    flux_data = defaultdict(list)
    flux_err = defaultdict(list) 
    sigma_data = defaultdict(list)
    sigma_err = defaultdict(list)
    shear_data = [defaultdict(list), defaultdict(list)]
    shear_err = [defaultdict(list), defaultdict(list)]
    centroid_data = [defaultdict(list), defaultdict(list)]
    centroid_err = [defaultdict(list), defaultdict(list)]

    with open(filename, 'rb') as f:
        dataset = pkl.load(f)

        for ishear, fdict in dataset.fluxes.items():
            for iflux, vals in fdict.items():
                x_data[ishear].append(np.mean(vals['Ideal']))
                x_err[ishear].append(np.std(vals['Ideal']))
                shear_diffs = [[], []]
                centroid_diffs = [[], []]

                flux_diffs = dataset.fluxes[ishear][iflux]['ADC'] - dataset.fluxes[ishear][iflux]['Ideal'] 
                sigma_diffs = dataset.sigmas[ishear][iflux]['ADC'] - dataset.sigmas[ishear][iflux]['Ideal'] 
                shear_diffs[0] = dataset.g1s[ishear][iflux]['ADC'] - dataset.g1s[ishear][iflux]['Ideal'] 
                shear_diffs[1] = dataset.g2s[ishear][iflux]['ADC'] - dataset.g2s[ishear][iflux]['Ideal'] 
                centroid_diffs[0] = dataset.centroid_xs[ishear][iflux]['ADC'] - dataset.centroid_xs[ishear][iflux]['Ideal'] 
                centroid_diffs[1]= dataset.centroid_ys[ishear][iflux]['ADC'] - dataset.centroid_ys[ishear][iflux]['Ideal'] 

                flux_data[ishear].append(np.mean(flux_diffs))
                sigma_data[ishear].append(np.mean(sigma_diffs))

                flux_err[ishear].append(np.std(flux_diffs))
                sigma_err[ishear].append(np.std(sigma_diffs))

                for i in range(2):
                    shear_data[i][ishear].append(np.mean(shear_diffs[i]))
                    centroid_data[i][ishear].append(np.mean(centroid_diffs[i])) 

                    shear_err[i][ishear].append(np.std(shear_diffs[i]))
                    centroid_err[i][ishear].append(np.std(centroid_diffs[i])) 

    data_list = [flux_data, sigma_data, shear_data, centroid_data]
    errs_list = [flux_err, sigma_err, shear_err, centroid_err]
    cutoff = -2

    for title, label, data, err in zip(titles, ylabels, data_list, errs_list):
        if title == 'Shear' or title == 'Centroid':
            fig, axs = plt.subplots(2, 1, figsize=(20, 20), sharex=True) 

            for l, ax, y, yerr in zip(label, axs.ravel(), data, err):
                for inshear, input_shear in enumerate(input_shears):
                    ax.errorbar(x_data[inshear][:cutoff], y[inshear][:cutoff], xerr=x_err[inshear][:cutoff], yerr=yerr[inshear][:cutoff],
                                linestyle='', label=r'$\gamma$ = ' + f'{input_shear[0]}+i{input_shear[1]}')
                    ax.legend(fontsize=12)
                    ax.set_ylabel(l, fontsize=18)
                    ax.grid(b=True)
            axs[1].set_xlabel('flux', fontsize=18)

        else:
            fig, ax = plt.subplots(figsize=(20, 20)) 

            for inshear, input_shear in enumerate(input_shears):
                #print(input_shear, x_data[inshear][:cutoff], data[inshear][:cutoff], err[inshear][:cutoff])
                ax.errorbar(x_data[inshear][:cutoff], data[inshear][:cutoff], xerr=x_err[inshear][:cutoff], yerr=err[inshear][:cutoff],
                            linestyle='', label=r'$\gamma$ = ' + f'{input_shear[0]}+i{input_shear[1]}')
                ax.legend(fontsize=12)
                ax.set_ylabel(label, fontsize=18)
                ax.grid(b=True)

            ax.set_xlabel('flux', fontsize=18)

        fig.suptitle(f'ADC - Ideal {title} vs Ideal Flux', fontsize=24)

        figname = f'{title.lower()}_diffs_v_flux.png'
        fig.savefig(os.path.join(save_to, figname))
        print(f'Wrote {os.path.join(save_to, figname)}')
 
def get_real_probs():
    dataset_file = '/gpfs/slac/lsst/fs1/u/adriansh/analysis/adc/datasets/bit/12781_R30_S20_bit_dataset.pkl'
    amp = 'C00'
    with open(dataset_file, 'rb') as f:
        ds = pkl.load(f)
        meds = ds.amp_medians[amp]
        probs = ds.bit_probs[amp]
    return meds, probs

def interpolate_probs():
    xpad = 2**9 # low signal flats have high variance bit4 probs
    meds, probs = get_real_probs()
    mx = meds - np.min(meds) - xpad
    ip = interpolate.interp1d(mx, probs, axis=0, kind='cubic')
    xend = np.max(mx)
    x = np.arange(0, xend, 32)
    return x, ip(x)
    
def make_dnl_bins():
    x, inprobs = interpolate_probs()
    bin_edges = np.array([])
    lp = len(inprobs[0])
    for xx, probs in zip(x, inprobs):
        temp_edges = [0]
        bin_ws = np.power(2, np.arange(1, lp+1)) * probs
        for i in range(lp):
            new_edges = []
            for edge in temp_edges:
                new_edges.append(edge + bin_ws[-i])
            temp_edges = temp_edges + new_edges
            temp_edges.sort()
        temp_edges = np.array(temp_edges) + xx
        bin_edges = np.concatenate((bin_edges, temp_edges))
    return bin_edges

def plot_for_steve():
    filename = 'analysis/adc/datasets/galsim/galsim_dnl_dataset.pkl'
    save_to = 'analysis/adc/plots/galsim/'

    with open(filename, 'rb') as f:
        dataset = pkl.load(f)
        ideal_mus = dataset.means['Ideal']
        ideal_sigs = dataset.sigmas['Ideal']
        ideal_fluxes = dataset.fluxes['Ideal']
        ideal_g1s = dataset.g1s['Ideal']
        ideal_g2s = dataset.g2s['Ideal']
        ideal_cxs = dataset.centroid_xs['Ideal']
        ideal_cys = dataset.centroid_ys['Ideal']

        adc_mus = dataset.means['ADC']
        adc_sigs = dataset.sigmas['ADC']
        adc_fluxes = dataset.fluxes['ADC']
        adc_g1s = dataset.g1s['ADC']
        adc_g2s = dataset.g2s['ADC']
        adc_cxs = dataset.centroid_xs['ADC']
        adc_cys = dataset.centroid_ys['ADC']

        titles = ['Sigma', 'Flux', 'Shear', 'Centroid']

        label_list = ['', '', ['g1', 'g2'], ['cx', 'cy']]

        data_list = [adc_sigs - ideal_sigs, adc_fluxes - ideal_fluxes,
                    [adc_g1s - ideal_g1s, adc_g2s - ideal_g2s],
                    [adc_cxs - ideal_cxs, adc_cys - ideal_cys]]

        fig, axs = plt.subplots(2, 2, figsize=(20, 20), sharex=True)
        fig.suptitle('ADC - Ideal vs Ideal Flux', fontsize=24)

        for ax, datas, labels, title in zip(axs.ravel(), data_list, label_list, titles):
            ax.set_title(title, fontsize=18)
            ax.grid(b=True)
            if len(datas) == 2:
                for data, label in zip(datas, labels):
                    ax.scatter(ideal_fluxes, data, label=label)
                    ax.legend(fontsize=12)
            else:
                ax.scatter(ideal_fluxes, datas)
        figname = 'diffs_v_flux.png'
        fig.savefig(os.path.join(save_to, figname))
        print(f'Wrote {os.path.join(save_to, figname)}') 

def plot_correction():
    filename = 'analysis/adc/datasets/galsim/galsim_dnl_dataset.pkl'
    save_to = 'analysis/adc/plots/galsim/'

    with open(filename, 'rb') as f:
        dataset = pkl.load(f)
        ideal_mus = dataset.means['Ideal']
        ideal_sigs = dataset.sigmas['Ideal']
        ideal_fluxes = dataset.fluxes['Ideal']
        ideal_g1s = dataset.g1s['Ideal']
        ideal_g2s = dataset.g2s['Ideal']
        ideal_cxs = dataset.centroid_xs['Ideal']
        ideal_cys = dataset.centroid_ys['Ideal']

        label_list = ['', '', ['g1', 'g2'], ['cx', 'cy']]

        data_dict = dict(Sigma=[dataset.sigmas['ADC']-ideal_sigs, dataset.sigmas['Corrected']-ideal_sigs],
                         Flux=[dataset.fluxes['ADC']-ideal_fluxes, dataset.fluxes['Corrected']-ideal_fluxes],
                         Shear=[[dataset.g1s['ADC']-ideal_g1s, dataset.g1s['Corrected']-ideal_g1s],
                                [dataset.g2s['ADC']-ideal_g2s, dataset.g2s['Corrected']-ideal_g2s]],
                         Centroid=[[dataset.centroid_xs['ADC']-ideal_cxs, dataset.centroid_xs['Corrected']-ideal_cxs],  
                                   [dataset.centroid_ys['ADC']-ideal_cys, dataset.centroid_ys['Corrected']-ideal_cys]])

        fig, axs = plt.subplots(2, 2, figsize=(20, 20))
        fig.suptitle('Corrected - Ideal vs Uncorrected - Ideal Measurements', fontsize=24)

        for ax, labels, (title, data_list) in zip(axs.ravel(), label_list, data_dict.items()):
            ax.set_title(title, fontsize=18)
            ax.set_xlabel('Uncorrected', fontsize=15)
            ax.set_ylabel('Corrected', fontsize=15)
            ax.tick_params(labelsize=15)
            ax.grid(b=True)
            #plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
            ax.ticklabel_format(style='sci', scilimits=(0,0))

            if title == 'Shear' or title == 'Centroid':
                for data, label in zip(data_list, labels):
                    x = data[0]
                    y = data[1]
                    ax.scatter(x, y, label=label)
                    ax.legend(fontsize=12)
            else:
                x = data_list[0]
                y = data_list[1]
                ax.scatter(x, y)

    figname = 'corrected_v_uncorrected.png'
    fig.savefig(os.path.join(save_to, figname))
    print(f'Wrote {os.path.join(save_to, figname)}')
 
def plot_dataset():
    filename = 'analysis/adc/datasets/galsim/galsim_dnl_dataset.pkl'
    save_to = 'analysis/adc/plots/galsim/'
    mean_bins = 50 #np.linspace(0.7, 0.8, 51)
    sigma_bins = 50 #np.linspace(1.5e-3, 4.5e-3, 51)
    flux_bins = 50 #np.linspace(20, 50, 51)
    shear_bins = 50 #np.linspace(-1.e-3, 1e-3, 51)
    centroid_bins = 50 #np.linspace(-1.5e-3, 1.5e-3, 51)

    with open(filename, 'rb') as f:
        dataset = pkl.load(f)
        ax_labels = ['Mean', 'Sigma', 'Flux', 'Shear', 'Centroid']

        hist_kwargs = [dict(bins=mean_bins),
                       dict(bins=sigma_bins),
                       dict(bins=flux_bins),
                       dict(label=[f'g1', 
                                   f'g2'],
                            bins=shear_bins, fill=False, linewidth=3, 
                            histtype='step', stacked=False),
                       dict(label=[f'x',
                                   f'y'],  
                            bins=centroid_bins, fill=False, linewidth=3,
                            histtype='step', stacked=False)]

        fig, axs = plt.subplots(5, 5, figsize=(20, 20), sharex=False)

        for ishear, fdict in dataset.means.items():
            input_shear = input_shears[ishear]

            for input_flux, means in fdict.items():
                for ax in axs.ravel():
                    ax.cla()
                ideal_mus = means['Ideal']
                ideal_sigs = dataset.sigmas[ishear][input_flux]['Ideal']
                ideal_fluxes = dataset.fluxes[ishear][input_flux]['Ideal']
                ideal_g1s = dataset.g1s[ishear][input_flux]['Ideal']
                ideal_g2s = dataset.g2s[ishear][input_flux]['Ideal']
                ideal_cxs = dataset.centroid_xs[ishear][input_flux]['Ideal']
                ideal_cys = dataset.centroid_ys[ishear][input_flux]['Ideal']

                ideal_mu = np.mean(ideal_mus)
                ideal_sig = np.mean(ideal_sigs)
                ideal_flux = np.mean(ideal_fluxes)
                ideal_g1 = ideal_g1s.mean()
                ideal_g2 = ideal_g2s.mean()
                ideal_cx = ideal_cxs.mean()
                ideal_cy = ideal_cys.mean() 

                hist_titles = [r'$\mu_i$' + f' = {ideal_mu:.3E}', r'$\sigma_i$' + f' = {ideal_sig:.3E}', 
                               r'Flux$_i$' + f' = {ideal_flux:.3E}', 
                               r'$\gamma_{1,i}$' + f' = {ideal_g1:.3E}, ' + r'$\gamma_{2,i}$' + f' = {ideal_g2:.3E}, ',
                               r'$cx_i$' + f' = {ideal_cx:.3E}, ' + r'$cy_i$' + f' = {ideal_cy:.3E}, ']

                for key in ['ADC']:#, 'Corrected']:
                    data = [dataset.means[ishear][input_flux][key] - ideal_mus, 
                            dataset.sigmas[ishear][input_flux][key] - ideal_sigs,
                            dataset.fluxes[ishear][input_flux][key] - ideal_fluxes, 
                            [dataset.g1s[ishear][input_flux][key] - ideal_g1s, 
                             dataset.g2s[ishear][input_flux][key] - ideal_g2s],
                            [dataset.centroid_xs[ishear][input_flux][key] - ideal_cxs, 
                             dataset.centroid_ys[ishear][input_flux][key] - ideal_cys]]


                for i in range(5):
                    axs[i, 0].set_ylabel(ax_labels[i], ha='right', fontsize=18)
                    axs[-1, i].set_xlabel(ax_labels[i], fontsize=18)

                    axs[i, i].hist(data[i], **hist_kwargs[i])
                    axs[i, i].set_title(hist_titles[i], fontsize=18)
                    axs[i, i].legend(fontsize=12)

                    if i < 4:
                        axs[i, i].set_xticklabels([])
                    else:
                        plt.setp(axs[i, i].get_xticklabels(), rotation=30, ha='right')

                    for j in range(0, i):
                        if i < 4:
                            axs[i, j].set_xticklabels([])
                        
                        if j > 0:
                            axs[i, j].set_yticklabels([])

                        if j == 1:
                            plt.setp(axs[i, j].get_xticklabels(), rotation=30, ha='right')

                        if i == 3:
                            axs[i, j].scatter(data[j], data[i][0], label=f'g1', color='tab:blue', s=10)
                            axs[i, j].scatter(data[j], data[i][1], label=f'g2', color='tab:orange', s=10)
                        elif i == 4 and j < 3:
                            axs[i, j].scatter(data[j], data[i][0], label=f'cx', color='tab:blue', s=10)
                            axs[i, j].scatter(data[j], data[i][1], label=f'cy', color='tab:orange', s=10)
                        elif i == 4 and j == 3:
                            axs[i, j].scatter(data[j][0], data[i][0], label=f'g1, cx', color='tab:blue', s=10)
                            axs[i, j].scatter(data[j][0], data[i][1], label=f'g1, cy', color='tab:green', s=10)
                            axs[i, j].scatter(data[j][1], data[i][0], label=f'g2, cx', color='tab:orange', s=10)
                            axs[i, j].scatter(data[j][1], data[i][1], label=f'g2, cy', color='tab:purple', s=10)
                            axs[i, j].legend(fontsize=12)
                            plt.setp(axs[i, j].get_xticklabels(), rotation=30, ha='right')
                        else:
                            axs[i, j].scatter(data[j], data[i], s=10)
                        axs[i, j].grid(b=True)

                    # hide upper off-diagonals
                    for k in range(i+1, 5):
                        axs[i, k].set_frame_on(False)
                        axs[i, k].set_xticks([])
                        axs[i, k].set_yticks([])

                fig.suptitle(f'{key} - Ideal for Input Shear = {input_shear}, Input Flux = {input_flux}', fontsize=26)
                gam1 = input_shear[0]
                gam2 = input_shear[1]
                fig.savefig(os.path.join(save_to, f'{key}_ideal_diff_{gam1}_{gam2}_{input_flux}.png'))
                print(f'Wrote {os.path.join(save_to, f"{key}_ideal_diff_{gam1}_{gam2}_{input_flux}.png")}') 

def debug_scratch():
    fname = 'analysis/adc/datasets/galsim/galsim_dnl_dataset.pkl'
    n_shear = 2
    #shear = (0, 0.25)
    iflux = 1000000
    with open(fname, 'rb') as f:
        ds = pkl.load(f)
        fluxes = ds.fluxes[n_shear][iflux]['Ideal']
        diffs = ds.fluxes[n_shear][iflux]['ADC'] - ds.fluxes[n_shear][iflux]['Ideal']
    print(diffs)#np.where(diffs<-100))

input_fluxes = [10000, 50000, 100000, int(7e5), int(1.4e6)]
input_shears = [(0, 0), (0.25, 0), (0, 0.25), (-0.25, 0)]
#make_star_image(10000, 0, 0)
#write_star_images(input_fluxes, input_shears, n_ims=500)

#cursed_images = [185]
#fname = '/gpfs/slac/lsst/fs1/u/adriansh/analysis/adc/galsim/star_images/star_images.pkl'
#im_dict = read_star_images(fname)
#make_dataset(im_dict)
#debug_scratch()
#plot_star_image(im_dict, input_shears[2], input_fluxes[-2], 185)

#plot_dataset()
plot_vals(input_shears)
#plot_correction()
#make_dnl_bins()

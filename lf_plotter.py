import lf_model, lf_optimizer
import emcee
import numpy as np
import corner
import utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import re
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from scipy.stats import norm

class LFPlotter:
    def __init__(self, directories, ModelClass=lf_model.FiducialCLF, names=None, f_esc=0.2, manual_params=None, dc=True):
        utils.my_mpl()
        if names is None:
            names = directories
        self.names = names
        self.directories = directories
        self.ModelClass = utils.to_array(ModelClass, extend=len(directories))
        self.mcmc_models = []
        self.mcmc_readers = []
        self.models = []
        dc = utils.to_array(dc, extend=len(directories))
        manual_params = utils.to_array(manual_params, extend=len(directories))
        f_esc = utils.to_array(f_esc, extend=len(directories))
        i = 0
        for k, d in enumerate(directories):
            MClass = self.ModelClass[k]
            try:
                self.mcmc_readers.append(
                    emcee.backends.HDFBackend(d + '/mcmc.h5'))
            except:
                pass
            if manual_params[k] is None:
                try:
                    params = np.load(d + '/best.npy')
                except FileNotFoundError:
                    params = np.zeros(MClass.N_PARAMS)
            else:
                params = manual_params[k]
            
            #print(names, f_esc, params, i)
            self.models.append(MClass(
                d + '/meas.npz', params, name=names[i], f_esc=f_esc[i], dc=dc[i]))
            i += 1

    def mcmc_results(self, burn_in=5000, which_model=0, sigmas=[-5,-4,-3,-2,-1,0,1,2,3,4,5], supress_print=False, for_latex=False, decimals=12):
        reader = self.mcmc_readers[which_model]
        model = self.models[which_model]
        samples = reader.get_chain(discard=burn_in, flat=True)
        labels = model.param_names
        # if not supress_print:
        #     print(labels)
        percentiles =  [100*norm.cdf(sigma) for sigma in sigmas]
        sig_array = []
        for i in range(model.N_PARAMS):
            mean = np.percentile(samples[:,i], 50)
            values = np.percentile(samples[:,i], percentiles)
            diff = np.round(values-mean[()], decimals)
            values = np.round(values, decimals)
            sig_array.append([labels[i], list(zip(sigmas, values, diff))])
            if not supress_print:
                if for_latex:
                    if len(sigmas) != 3:
                        raise NotImplementedError
                    print(' ')
                    print(fr'{labels[i]}$={values[1]}\pm^{{{diff[2]}}}_{{{-1*diff[0]}}}$')
                    print(' ')
                else:
                    print(labels[i], list(zip(sigmas, values, diff)))
        return sig_array
    
    def initialize_optimize(self, burn_in=15000, which_model=0, maxfev=10000, maxiter=10000, x0=None):
        model_class = self.ModelClass[which_model]
        model_dir = self.directories[which_model]
        if x0 is None:
            results = self.mcmc_results(burn_in=burn_in, which_model=which_model, sigmas=[0], supress_print=True, decimals=12)
            x0 = [elem[1][0][1] for elem in results]
            print('best_fit', x0)
        lf_optimizer.run_optimize(directories=[model_dir], x0=x0, ModelClass=model_class, maxfev=maxfev, maxiter=maxiter)

    def save_fig(self, fn, fig_dir='figs/', png=False):
        if png:
            plt.savefig('{}/{}.png'.format(fig_dir, fn))
        else:
            plt.savefig('{}/{}.pdf'.format(fig_dir, fn))

    def populate_mcmc_models(self, burn_in=15000, skip=5003):
        if len(self.mcmc_models) == 0:
            for i, d in enumerate(self.directories):
                params_list = self.mcmc_readers[0].get_blobs()[burn_in:]
                n_attempts = params_list.shape[0] * params_list.shape[1]
                n_params = params_list.shape[2]
                params_list = params_list.reshape(n_attempts, n_params)
                for k in range(0, n_attempts, skip):
                    self.mcmc_models.append(self.ModelClass[0](
                        d + '/meas.npz', params_list[k], name=self.names[0]))

    def plot_bands_M(self, ax, M_vals, func, args, sigmas=[1, 2, 3], burn_in=15000, skip=5003):
        self.populate_mcmc_models(burn_in=burn_in, skip=skip)
        runs = []
        for model in self.mcmc_models:
            runs.append([func(model, M, *args) for M in M_vals])
        band_handles = []
        band_labels = []
        for sigma in sigmas:
            p_val = utils.sigma_to_p_value(sigma)
            above = np.percentile(runs, 100*(1-p_val), axis=0)
            below = np.percentile(runs, 100*p_val, axis=0)
            bh = ax.fill_between(M_vals, above, below,
                                 alpha=0.1, color='k', zorder=-1)
            band_handles.append(bh)
            band_labels.append(r'$\pm${}$\sigma$'.format(sigma))
        band_handles[0] = (band_handles[0], band_handles[1], band_handles[2])
        band_handles[1] = (band_handles[1], band_handles[2])
        band_legend = ax.legend(band_handles, band_labels, loc='upper left')
        ax.add_artist(band_legend)

    def loop_over_models(self, function, **kwargs):
        for i, mod in enumerate(self.models):
            print(mod.name)
            function(which_model=i, **kwargs)

    def mcmc_taus(self):
        for i, reader in enumerate(self.mcmc_readers):
            print(self.models[i].name)
            print(reader.get_chain().shape)
            try:
                tau = reader.get_autocorr_time()
                print("autocorrelation times:", tau)
                print('max of 50 times tau', max(tau*50))
            except emcee.autocorr.AutocorrError as e:
                print(e)

    def plot_cor_matrix(self, burn_in=15000, which_model=0, fontsize=20):
        plt.figure()
        reader = self.mcmc_readers[which_model]
        samples = reader.get_chain(flat=True, discard=burn_in)
        cov = np.corrcoef(samples.T)
        print(cov)
        plt.grid(False)
        plt.imshow(cov, vmin=-1, vmax=1)
        fsize=fontsize
        labels = self.models[which_model].param_names
        plt.xticks(ticks=plt.xticks()[0][1:-1], labels=labels, fontsize=fsize)
        plt.yticks(ticks=plt.yticks()[0][1:-1], labels=labels, fontsize=fsize)


        for i in range(cov.shape[0]):
            for j in range(cov.shape[1]):
                rounded_cov = round(cov[i, j],1)
                lab = str(rounded_cov) if rounded_cov != -0.0 else '0.0'
                plt.annotate(lab, xy=(j, i),
                            ha='center', va='center', color='white', fontsize=28)

        colorbar = plt.colorbar()
        ticks = colorbar.get_ticks().tolist()  # Get the current tick locations
        ticks.append(-1)  # Add -1 to the list of ticks
        colorbar.set_ticks(ticks)  # Set the new tick locations
        colorbar.set_ticklabels(ticks)  # Set the tick labels
        plt.title('Correlation Matrix')
        plt.tight_layout()

    def plot_par_histogram(self, burn_in=15000, which_model=0, which_param=-1, show_best=True, ):
        reader = self.mcmc_readers[which_model]
        model = self.models[which_model]
        samples = reader.get_chain(discard=burn_in, flat=True)
        plt.hist(samples[:, which_param], 100, color='k', histtype='step')
        label = model.param_names[which_param]
        plt.xlabel(rf'{label}')
        plt.ylabel(rf"$p(${label}$)$")

    
    def plot_corner(self, burn_in=15000, which_model=0, show_best=True, walkers=False, sigmas=[-3,-2,-1,0,1,2,3], levels2D=None, title_fmt='.3g'):
        mpl.style.use('default')
        plt.rc('font', family='serif', size=14)
        reader = self.mcmc_readers[which_model]
        model = self.models[which_model]
        samples = reader.get_chain(discard=burn_in, flat=True)
        labels = model.param_names
        sig_array = self.mcmc_results(burn_in=burn_in, which_model=which_model, sigmas=sigmas)
        quants = [utils.ONE_SIGMA, 0.5, 1-utils.ONE_SIGMA]
        label_kwargs = dict(fontsize=20)
        fig = corner.corner(samples, show_titles=True, labels=labels,
                            quantiles=quants, label_kwargs=label_kwargs, levels=levels2D, title_fmt=title_fmt) #axes_scale='log')
        params = model.params
        if show_best:
            corner.overplot_lines(fig, params, lw=3)
            corner.overplot_points(
                fig, params[None], marker='s', markeredgecolor='w', markersize=8, markeredgewidth=1.5)

        if walkers:
            try:
                tau = reader.get_autocorr_time()
                print("autocorrelation times:", tau)
            except emcee.autocorr.AutocorrError as e:
                print(e)
            _, axes = plt.subplots(model.N_PARAMS, figsize=(
                16, 4*model.N_PARAMS), sharex=True)
            all_samples = reader.get_chain()
            for i in range(model.N_PARAMS):
                ax = axes[i]
                ax.plot(all_samples[:, :, i], 'k', alpha=0.05)
                ax.set_xlim(0, len(all_samples))
                ax.set_ylabel(labels[i])
                ax.yaxis.set_label_coords(-0.1, 0.5)
                axes[-1].set_xlabel("step number")
        #plt.tight_layout()
        utils.my_mpl()
        return sig_array
    
    def plot_chi(self, z_vals, n_col=1, ax_lims=[None, None], legend_fontsize=None, figsize=None, which_model=0):
        mod = self.models[which_model]
        chi_dict = mod.chi_sq_of_model(at_each_point=True)
        mod_z = mod.meas.z_vals
        mod_mags = mod.meas.mags
        plt.figure(figsize=figsize)
        for z in z_vals:
            mUV = mod_mags[mod_z==z]
            plt.plot(mUV, chi_dict[z], label=f'z={z}')
        plt.legend(fontsize=legend_fontsize)

    def plot_errbars_helper(self, model, ax, z, unique_sources):
        colors = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'brown', 'magenta', 'cyan']
        def paint(source):
            idx = np.argwhere(unique_sources == source)[0][0]
            return colors[idx]
        for z_vals, mags, phis, sm, sp, sources in model.meas.zipped:
            if z_vals == z:
                capsize = 5
                uplims = False
                no_lower_cap = False
                zorder = None
                lolims = False
                marker = 'o'
                label = sources
                if sm == 0 or phis - sm <= 0:
                    uplims = True
                    if sm == 0:
                        phis = sp
                        sm = 0.9*sp
                        marker=''
                        label=None #FIXME: THIS IS DANGEROUS SINCE ANY SET THAT IS ONLY UPPER ERROR BARS IS LOST!
                    elif phis - sm <= 0:
                        sm = 0.9*phis if phis > 2e-6 else 0.5*phis
                        _, caplines, _ = ax.errorbar(mags, phis, yerr=[[sm], [sp]], ls='', marker=marker, capsize=capsize, color=paint(sources), label=label, uplims=True, lw=3)
                        caplines[1].set_marker('')
                        uplims=False
                        zorder=99
                        no_lower_cap = True
                else:
                    _, caplines, _ = ax.errorbar(mags, phis, yerr=[[sm], [sp]], ls='', marker=marker, capsize=capsize, color=paint(sources), label=label, uplims=uplims, lw=3, lolims=lolims, zorder=zorder)
                    if no_lower_cap:
                        caplines[0].set_marker('')


    def plot_fits(self, n_col=1, use_bands=False, eval_mags=None, eval_z=None, ax_lims=[[-25.5, -12.5], [1e-7,7.5]], second_ylim=None, legend_fontsize=None, figsize=None, plot_slope=False, webb=False, two_sigma=False, model_colors=None, model_ls=None, handlelength=None):
        unique_z = self.models[0].meas.unique_z if eval_z is None else eval_z
        n_row = int(np.ceil(len(unique_z)/n_col))
        figsize = figsize if figsize is not None else [16, 5*n_row]
        sharey = 'all'
        if second_ylim is not None:
            sharey = 'row'
        fig, axes = plt.subplots(
            n_row, n_col, sharex='all', sharey=sharey, figsize=figsize)
        axs = np.ravel(axes, order='F')
        if n_col % 2 == 1:
            axs = np.ravel(axes, order='C')
        plt.subplots_adjust(hspace=0, wspace=0)
        colors = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'brown', 'magenta', 'cyan']
        markers = 6*['o']
        if webb:
            markers = ['o', 'o', 's', '^', 'v', 'D', 'p', 'h']

        unique_sources = []
        for model in self.models:
            unique_sources = np.union1d(
                unique_sources, list(set(model.meas.sources)))
        
        def paint(source):
            idx = np.argwhere(unique_sources == source)[0][0]
            return colors[idx]

        def shape(source):
            idx = np.argwhere(unique_sources == source)[0][0]
            return markers[idx]
        for i, z in enumerate(unique_z):
            ax = axs[i]
            mags = []
            for model in self.models:
                mags = np.union1d(mags, [m for (m, z_vals)
                                         in model.meas.mags_and_zs if z_vals == z])
            for k, model in enumerate(self.models):
                if eval_mags is not None:
                    phi = [model.phi_m(m=eval_mag, z=z) for eval_mag in eval_mags]
                    if plot_slope:
                        ax.plot(0.5*(eval_mags[1:]+eval_mags[:-1]), (-1/0.4)*np.diff(np.log10(phi))/np.diff(
                            eval_mags) - 1, label=model.name, ls=utils.LINESTYLE_ARR[k])
                    else:
                        if model_colors is None:
                            model_colors = len(self.models)*[None]
                        if model_ls is None:
                            model_ls = utils.LINESTYLE_ARR
                        ax.plot(eval_mags, phi, label=model.name,
                                ls=model_ls[k], color=model_colors[k])
                else:
                    if model_colors is None:
                        model_colors = len(self.models)*[None]
                    if model_ls is None:
                        model_ls = utils.LINESTYLE_ARR
                    mag_range = np.linspace(min(mags), max(mags), 100)
                    phi = [model.phi_m(mag, z) for mag in mag_range]
                    ax.plot(mag_range, phi, label=model.name,
                            ls=model_ls[k], color=model_colors[k])
            if use_bands:
                self.populate_mcmc_models()
                runs = []
                for model in self.mcmc_models:
                    runs.append([model.phi_m(mag, z) for mag in mags])
                for sigma in [1, 2, 3]:
                    p_val = utils.sigma_to_p_value(sigma)
                    above = np.percentile(runs, 100*(1-p_val), axis=0)
                    below = np.percentile(runs, 100*p_val, axis=0)
                    ax.fill_between(mags, above, below,
                                    alpha=0.1, color='k', zorder=-1)
                #self.populate_mcmc_models()
                #for model in self.mcmc_models:
                #    ax.plot(mags, [model.phi_m(mag, z)
                #                   for mag in mags], lw=1, color='C1', alpha=0.1, zorder=-1)
            for model in self.models:
                # TODO: THIS IS REALLY UGLY PLEASE FIND A BETTER WAY
                if plot_slope:
                    break
                for z_vals, mags, phis, sm, sp, sources in model.meas.zipped:
                    #print(sources)
                    if z_vals == z:
                        capsize = 5
                        uplims = False
                        no_lower_cap = False
                        zorder = None
                        lolims = False
                        marker = 'o' #shape(sources)
                        label = sources
                        if sm == 0 or phis - sm <= 0:
                            uplims = True
                            if sm == 0:
                                phis = sp
                                sm = 0.9*sp
                                marker=''
                                label=None #FIXME: THIS IS DANGEROUS SINCE ANY SET THAT IS ONLY UPPER ERROR BARS IS LOST!
                            elif phis - sm <= 0:
                                sm = 0.9*phis if phis > 2e-6 else 0.5*phis
                                _, caplines, _ = ax.errorbar(mags, phis, yerr=[[sm], [sp]], ls='', marker=marker, capsize=capsize, color=paint(sources), label=label, uplims=True, lw=2.5)
                                caplines[1].set_marker('')
                                uplims=False
                                #lolims=True
                                zorder=99
                                no_lower_cap = True
                                #ax.plot([x[i], x[i]], [y[i] - y_lower_err[i], y[i] + y_upper_err[i]], color='black', linewidth=1)
                        if (sources == 'Harikane+23a' or sources == 'Harikane+23') and z == 16:
                            xerr = 1.5
                            ax.errorbar(mags, phis, yerr=[[sm], [sp]], xerr=xerr, ls='', marker=marker, capsize=capsize, color=paint(sources), label=label, uplims=uplims, lw=2.5)
                        else:
                            _, caplines, _ = ax.errorbar(mags, phis, yerr=[[sm], [sp]], ls='', marker=marker, capsize=capsize, color=paint(sources), label=label, uplims=uplims, lw=2.5, lolims=lolims, zorder=zorder)
                            if no_lower_cap:
                                caplines[0].set_marker('')
                if two_sigma:
                    [ax.errorbar(mags.astype(float), phis.astype(float), yerr=[[2*sm.astype(float)], [2*sp.astype(float)]], ls='', marker='', capsize=10, color=paint(
                    sources), label=sources + r' $2\sigma$', alpha=0.2) for z_vals, mags, phis, sm, sp, sources in model.meas.zipped if z_vals == z]

            if not plot_slope:
                ax.set_yscale('log')
            if i < n_row and n_col % 2 == 0 or i % n_col == 0 and n_col % 2 == 1:
                ax.set_ylabel(r'$\phi(M_{\rm UV})$ [Mpc$^{-3}$ mag$^{-1}$]')
            ax.set_xlabel(r'$M_{\rm UV}$ [mag]')
            title = r'$z=$' + str(int(z))
            if z > 10:
                title = r'$z\sim$' + str(int(z))
            
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if i == 0:
                ax.text(0.65, 0.85, title, horizontalalignment='center',
                    transform=ax.transAxes, size='xx-large')
                ax.legend(list(by_label.values())[:], list(by_label.keys())[:],
                      loc='upper left', prop={'size': legend_fontsize}, handlelength=handlelength)
            elif i == 1:
                ax.text(0.57, 0.85, title, horizontalalignment='center',
                    transform=ax.transAxes, size='xx-large')
                ax.legend(list(by_label.values())[-2:], list(by_label.keys())[-2:],
                      loc='upper left', prop={'size': legend_fontsize})
            else:
                ax.text(0.5, 0.85, title, horizontalalignment='center',
                    transform=ax.transAxes, size='xx-large')
            #ax.set_xticks([-24, -22, -20, -18, -16, -14])
            if ax_lims[0] is not None or ax_lims[1] is not None:
                ax.set_xlim(ax_lims[0])
                ax.set_ylim(ax_lims[1])
                if second_ylim and i >= n_col:
                    ax.set_ylim(second_ylim)
        plt.tight_layout()
        plt.subplots_adjust(wspace=0, hspace=0)

    def plot_fits_and_reldif(self, n_col=1, eval_mags=None, eval_z=None, xlim=None, ylims=[None], legend_fontsize=None, figsize=None, hwspace=[0.02,0.02], hr=[3,1], xticks=None):
        unique_z = self.models[0].meas.unique_z if eval_z is None else eval_z
        n_row = 2*int(np.ceil(len(unique_z)/n_col))
        figsize = figsize if figsize is not None else [16, 2*n_row]
        hr = int(n_row/2)*hr
        fig, axes = plt.subplots(
            n_row, n_col, sharex='all', sharey='row', figsize=figsize, gridspec_kw={'height_ratios':hr})
        unique_sources = []
        for model in self.models:
            unique_sources = np.union1d(
                unique_sources, list(set(model.meas.sources)))
        ylims = utils.to_array(ylims, extend=int(2*n_row/(2*len(ylims))))
        for i, z in enumerate(unique_z):
            col = i % n_col
            row = int(2*np.floor(i/n_col))
            ax = axes[row][col]
            reldif_ax = axes[row+1][col]
            mags = []
            for model in self.models:
                mags = np.union1d(mags, [m for (m, z_vals)
                                         in model.meas.mags_and_zs if z_vals == z])
            
            for k, model in enumerate(self.models):
                if eval_mags is not None:
                    phi = [model.phi_m(m=eval_mag, z=z) for eval_mag in eval_mags]
                    ax.plot(eval_mags, phi, label=model.name, ls=utils.LINESTYLE_ARR[k])
                    if k >= 0:
                        default_model = self.models[0]
                        default_phi = [default_model.phi_m(m=eval_mag, z=z) for eval_mag in eval_mags]
                        reldif_ax.plot(eval_mags, np.divide(phi,default_phi), label=model.name, ls=utils.LINESTYLE_ARR[k])
                else:
                    mag_range = np.linspace(min(mags), max(mags), 100)
                    phi = [model.phi_m(mag, z) for mag in mag_range]
                    ax.plot(mag_range, phi, label=model.name, ls=utils.LINESTYLE_ARR[k])
                    reldif_ax.plot(mag_range, phi, label=model.name, ls=utils.LINESTYLE_ARR[k])
            for model in self.models:
                self.plot_errbars_helper(model, ax, z, unique_sources)
            
            
            ax.set_yscale('log')
            if i < n_row and n_col % 2 == 0 or i % n_col == 0 and n_col % 2 == 1:
                ax.set_ylabel(r'$\phi(M_{\rm UV})$ [Mpc$^{-3}$ mag$^{-1}$]')
                reldif_ax.set_ylabel(r'$\phi/\phi_{\rm CDM}$')
            if 2*np.floor(i/n_col)+2 == n_row:
                reldif_ax.set_xlabel(r'$M_{\rm UV}$ [mag]')
                reldif_ax.set_xticks(xticks)
            title = r'$z=$' + str(int(z))
            if z > 10:
                title = r'$z\sim$' + str(int(z))
            ax.text(0.35, 0.80, title, horizontalalignment='center',
                    transform=ax.transAxes, size='xx-large')
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            if i==0:
                ax.legend(by_label.values(), by_label.keys(), prop={'size': legend_fontsize}, loc='lower right')
            ax.set_xlim(xlim)
            ax.set_ylim(ylims[int(row/2)])
            reldif_ax.set_ylim([None, 1.1])
        plt.tight_layout()
        plt.subplots_adjust(hspace=hwspace[0], wspace=hwspace[1])
        

    def ax_helper(self, ax, ax_scale, ax_lab, margins=[0.1, 0.05], ax_lims=[None, None], legend_args={}):
        ax.set_xscale(ax_scale[0])
        ax.set_yscale(ax_scale[1])
        ax.set_xlabel(ax_lab[0])
        ax.set_ylabel(ax_lab[1])
        ax.set_xlim(ax_lims[0])
        ax.set_ylim(ax_lims[1])
        ax.margins(*margins)
        ax.legend(**legend_args)
        plt.tight_layout()

    def kwargs_func_vs_M(self, model, z, z_num, model_num, len_z, model_colors=None):
        if model_colors is None:
            kw = {'color': 'C{}'.format(model_num)}
        else:
            print(len(model_colors), model_num)
            kw = {'color': model_colors[model_num]}
        if len_z > 1:
            if z_num < 1:
                kw.update({'label': '{}'.format(model.name)})
            #kw.update({'ls': utils.LINESTYLE_ARR[z_num]})
        else:
            kw.update({'label': '{}'.format(model.name)})
            #kw.update({'ls': utils.LINESTYLE_ARR[model_num]})
        return kw
    
    def plot_f_tilde(self, ax, M_vals, model, z):
        M_stars = np.array([model.M_star_of_z_func(M_init=M, z_init=z)(6)[0] for M in M_vals])
        f_tilde = M_stars / ((utils.OMEGA_B/utils.OMEGA_M)*M_vals)
        ax.plot(M_vals, f_tilde, color='red', ls=':',)

    def plot_vs_M(self, z_vals, func, ax_scales, ax_labs, figsize=[12, 9], second_axis_func=None, use_bands=False, ax_lims=[None, None], M_vals=None, plot_helper=lambda ax: None, legend_args={}, model_colors=None, tilde=False, band_args={}):
        _, ax = plt.subplots(figsize=figsize)
        use_min_max_mass = M_vals is None
        had_tilde = tilde
        for i, z in enumerate(utils.to_array(z_vals)):
            for k, model in enumerate(self.models):
                if use_min_max_mass:
                    M_vals = np.logspace(
                        *np.log10(model.min_max_mass(z)), 1000)
                kw = self.kwargs_func_vs_M(model, z, i, k, len_z=len(z_vals), model_colors=model_colors)
                func_vals = [func(model, M, z) for M in M_vals]
                if k == 2:
                    ax.plot(M_vals, func_vals, **kw, zorder=-1)
                else:
                    ax.plot(M_vals, func_vals, **kw)
                if use_bands:
                    self.plot_bands_M(ax, M_vals, func, [z], **band_args)
                    use_bands = False  # ONLY BANDS AT FIRST REDSHIFT ?and model?
                if tilde:
                    self.plot_f_tilde(ax, M_vals, model, z)
                    tilde = False # ONLY \tilde f_\star AT FIRST REDSHIFT and model
        if second_axis_func is not None:
            second_axis_func(ax)
        if len(z_vals) > 1 and not had_tilde:
            legend_args.update({'handles':ax.get_legend_handles_labels()[0] + [mpl.lines.Line2D([], [], linestyle='-', color='k', label=f'z={z_vals[0]}'), mpl.lines.Line2D([], [], linestyle='--', color='k', label=f'z={z_vals[1]}')]})
        elif had_tilde:
            legend_args.update({'handles':ax.get_legend_handles_labels()[0] + [mpl.lines.Line2D([], [], linestyle='-', color='k', label=f'z={z_vals[0]}'), mpl.lines.Line2D([], [], linestyle='--', color='k', label=f'z={z_vals[1]}'), mpl.lines.Line2D([], [], linestyle=':', color='red', label=rf'$\tilde f_\star(z={z_vals[0]})$')]})
        plot_helper(ax)
        self.ax_helper(ax, ax_scales, ax_labs, ax_lims=ax_lims,
                       legend_args=legend_args)
        #plot_helper(ax)

    def plot_dn_dM(self, z_vals, figsize=[9, 6], ax_lims=[[None, None], [None, None]], M_vals=None):
        def func(model, M, z):
            return model.dn_dM(M, z)
        ax_scales = ['log', 'log']
        ax_labs = [r'$M$ [M$_\odot$]', r'$dn/dM$']
        self.plot_vs_M(z_vals, func, ax_scales, ax_labs, figsize, ax_lims=ax_lims, M_vals=M_vals)

    def plot_dn_dlnM(self, z_vals, figsize=[9, 6], ax_lims=[[None, None], [None, None]], M_vals=None, model_colors=None, legend_args={}):
        def func(model, M, z):
            return M * model.dn_dM(M, z)
        ax_scales = ['log', 'log']
        ax_labs = [r'$M$ [M$_\odot$]', r'${\rm d}n/{\rm d}lnM$ [Mpc$^{-3}$]']
        self.plot_vs_M(z_vals, func, ax_scales, ax_labs, figsize, ax_lims=ax_lims, M_vals=M_vals, model_colors=model_colors, legend_args=legend_args)

    def plot_cummulative_mf(self, z_vals, figsize=[9, 6], ax_lims=[[None, None], [None, None]], M_vals=None, model_colors=None, legend_args={}):
        def func(model, M, z):
            integrand = lambda M_prime: model.dn_dM(M_prime, z)
            return utils.trapz_integrate(integrand, M, 1e15, logspace=True)
        ax_scales = ['log', 'log']
        ax_labs = [r'$M$ [M$_\odot$]', r'$n(>M)$ [Mpc$^{-3}$]']
        self.plot_vs_M(z_vals, func, ax_scales, ax_labs, figsize, ax_lims=ax_lims, M_vals=M_vals, model_colors=model_colors, legend_args=legend_args)

    def plot_f_star(self, z_vals, figsize=[9, 6], bands=False, ax_lims=[[1e9, None], [1e-2, 0.3]], show_Ma=False, M_vals=None, scale=1, legend_args={'fontsize':16, 'loc':'lower center','ncol':2}, model_colors=None, tilde=False, band_args={}):
        def func(model, M, z):
            return scale * model.f_star(M, z)
        ax_scales = ['log', 'log']
        ax_labs = [r'$M$ [M$_\odot$]', r'$f_\star(M, z)$']
        if len(z_vals) == 1:
            ax_labs = [r'$M$ [M$_\odot$]', rf'$f_\star(M, z={z_vals[0]})$']
        def plot_helper(ax): return None
        if show_Ma:
            def plot_helper(ax):
                lo_x, lo_y = utils.load_csv(
                    'digitized_plots/Ma18_fig4/Ma18_fig4_z6_lower.csv')
                hi_x, hi_y = utils.load_csv(
                    'digitized_plots/Ma18_fig4/Ma18_fig4_z6_upper.csv')
                ax.plot(10**lo_x, 10**lo_y, color='red', ls='-.')
                ax.plot(10**hi_x, 10**hi_y, color='red',
                        ls='-.', label=r'Ma18 $1\sigma$')
                ax.legend(loc='lower center')
        self.plot_vs_M(z_vals, func, ax_scales, ax_labs,
                       figsize, use_bands=bands, ax_lims=ax_lims, plot_helper=plot_helper, M_vals=M_vals, legend_args=legend_args, model_colors=model_colors, tilde=tilde, band_args=band_args)

    def plot_sfr(self, z_vals, figsize=[9, 6], ax_lims=[[1e9, None], [1e-2, None]], M_vals=None):
        def func(model, M, z):
            return model.LFModel.sfr(M, z)
        ax_scales = ['log', 'log']
        ax_labs = [r'$M$ [M$_\odot$]', r'SFR$(M, z)$']
        self.plot_vs_M(z_vals, func, ax_scales, ax_labs,
                       figsize, ax_lims=ax_lims, M_vals=M_vals)

    def plot_LM(self, z_vals, figsize=[9, 6], M_vals=None, ax_lims=[[1e9, None], [None, None]],):
        def func(model, M, z):
            return model.m_c(M,z)
        ax_scales = ['log', 'linear']
        ax_labs = [r'$M$ [M$_\odot$]', r'$M_{\rm UV}$ [mag]']

        def sfr_axis(ax):
            def ax2_func(m): return np.log10(utils.KAPPA_UV * utils.m_to_L(m))
            def ax2_inv_func(sfr): return utils.L_to_m(10**sfr/utils.KAPPA_UV)
            ax2 = ax.secondary_yaxis(
                location='right', functions=(ax2_func, ax2_inv_func))
            ax2.set_ylabel(r'$\log_{10}$(SFR/[M$_\odot$ yr$^{-1}$])')
        self.plot_vs_M(z_vals, func, ax_scales, ax_labs,
                       figsize, second_axis_func=sfr_axis, M_vals=M_vals, ax_lims=ax_lims)

    def plot_dn_dot_ion_dlnM(self, z_vals, figsize=[9, 6], ax_lims=[[1e9, 1e13], [1e-2, None]], M_vals=None, legend_args={'fontsize':16, 'ncol':2, 'loc':'lower left'}):
        def func(model, M, z):
            return model.dn_dot_ion_dlnM(M,z)
        ax_scales = ['log', 'log']
        ax_labs = [r'$M$ [M$_\odot$]', r'${\rm d}\dot n_{\rm ion}/{\rm d}\ln(M)$']
        self.plot_vs_M(z_vals, func, ax_scales, ax_labs,
                       figsize, ax_lims=ax_lims, M_vals=M_vals, legend_args=legend_args)

    def plot_dlnn_dot_ion_dlnM(self, z_vals, figsize=[9, 6], ax_lims=[[1e9, None], [1e-2, None]], M_vals=None, legend_args={'fontsize':16, 'ncol':2, 'loc':'lower left'}):
        def func(model, M, z):
            return model.dlnn_dot_ion_dlnM(M, z)
        ax_scales = ['log', 'log']
        ax_labs = [r'$M$ [M$_\odot$]', r'${\rm d}\ln\dot n_{ion}/{\rm d}\ln(M)$']
        self.plot_vs_M(z_vals, func, ax_scales, ax_labs,
                       figsize, ax_lims=ax_lims, M_vals=M_vals, legend_args=legend_args)

    def kwargs_func_vs_z(self, model, model_num, func_num):
        kw = {}
        if func_num == 0:
            kw = {'label': '{}'.format(model.name)}
        kw.update({'color': 'C{}'.format(model_num)})
        kw.update({'ls': utils.LINESTYLE_ARR[func_num]})
        return kw
    
    def kwargs_func_vs_z_tau(self, model, model_num, func_num):
        kw = {}
        if func_num == 0:
            kw = {'label': '{}'.format(model.name)}
        kw.update({'color': 'C{}'.format(model_num)})
        kw.update({'ls': utils.LINESTYLE_ARR[func_num]})
        return kw
    
    def kwargs_func_vs_z2(self, model, model_num, func_num):
        colors = [0, 4, 9, 1, 2, 3, 5, 6, 7, 8]
        kw = {'label': '{}'.format(model.name)}
        kw.update({'color': 'C{}'.format(colors[model_num])})
        kw.update({'ls': utils.LINESTYLE_ARR[model_num]})
        return kw

    def hist_kwargs_func_vs_z(self, model, model_num, func_num):
        kw = {'label': '{}'.format(model.name)}
        #kw.update({'color': 'C{}'.format(model_num)})
        #kw.update({'marker': utils.MARKER_ARR[model_num]})
        #kw.update({'s': 100})
        return kw
    
    def sc_kwargs_func_vs_z(self, model, model_num, func_num):
        kw = {'label': '{}'.format(model.name)}
        kw.update({'color': 'C{}'.format(model_num)})
        kw.update({'marker': utils.MARKER_ARR[model_num]})
        kw.update({'s': 100})
        return kw

    def rho_kwargs_func_vs_z(self, model, model_num, func_num):
        kw = {}
        kw.update({'color': 'C{}'.format(model_num)})
        lab = model.name
        markers =['o', 'x', '+', 'v', '^', '<', '>', 's', 'p', 'h', 'd']
        if func_num == 0:
            kw.update({'ls': utils.LINESTYLE_ARR[model_num]})
            lab += ' Total'
        else:
            lab = 'Observed'
            #kw.update({'ls': 'none'})
            kw.update({'marker': markers[model_num]})
            kw.update({'s': 200})
        kw.update({'label': '{}'.format(lab)})
        return kw

    def plot_vs_z(self, z_range, func, ax_scales, ax_labs, figsize, margins=[0, 0.05], ax_lims=[None, None], plot_helper=lambda ax: None, sc_plot=False, hist_plot=False, kw_func=None, legend_args={}):
        if kw_func is None:
            kw_func = self.kwargs_func_vs_z
        _, ax = plt.subplots(figsize=figsize)
        if hist_plot:
           z_vals = np.arange(z_range[0], z_range[1]+1)
        else:
            sc_plot = utils.to_array(sc_plot, extend=len(utils.to_array(func)))
            z_vals = np.linspace(z_range[0], z_range[1], 10000)
        offset = -1/4
        for i, model in enumerate(self.models):
            for k, function in enumerate(utils.to_array(func)):
                model.__init__(model.meas_fn, model.params,
                               model.dc, model.name, model.f_esc)
                kw = kw_func(model, i, k)
                if hist_plot:#sc_plot[k]:
                    plt.bar(z_vals + offset, [function(model, z)
                                              for z in z_vals], zorder=99, width=1/4, **kw)
                    offset += 1/4
                elif sc_plot[k]:
                    z_bins = np.arange(5,11)                
                    ax.scatter(z_bins, [function(model, z) for z in z_bins], zorder=99, **kw)
                else:
                    ax.plot(z_vals, [function(model, z) for z in z_vals], **kw)
        self.ax_helper(ax, ax_scales, ax_labs,
                       margins=margins, ax_lims=ax_lims, legend_args=legend_args)
        plot_helper(ax)

    def plot_n_dot_ion(self, z_range, figsize=[12, 6]):
        def func(model, z):
            return model.n_dot_ion(z)
        ax_scales = ['linear', 'log']
        ax_labs = [r'$z$', r'$\langle \dot n_{ion}(z)\rangle$']
        self.plot_vs_z(z_range, func, ax_scales, ax_labs, figsize)

    def plot_rec_ratio(self, z_range, figsize=[12, 6]):
        def func(model, z):
            n = model.LFModel.n_dot_ion(z)
            x = model.LFModel.xi(z)
            t = model.LFModel.t_rec(z)
            return (x/t) / n
        ax_scales = ['linear', 'log']
        ax_labs = [r'$z$', r'$\dot (x_i(z)/t_{rec}(z))/n_{ion}(z)$']
        self.plot_vs_z(z_range, func, ax_scales, ax_labs, figsize)

    def plot_n_dot_ion_obs(self, z_range, figsize=[12, 6]):
        def func(model, z):
            a = model.n_dot_ion(z)
            b = model.n_dot_ion_obs_vs_tot(z)
            return a * b
        ax_scales = ['linear', 'log']
        ax_labs = [r'$z$', r'$\langle \dot n_{ion}(z)\rangle$']
        self.plot_vs_z(z_range, func, ax_scales, ax_labs, figsize)

    def plot_n_dot_ion_obs_vs_tot(self, z_range, figsize=[12, 6], hist_plot=True, legend_fontsize=12, ax_lims=[None, [0, 1]], y_size=None):
        def func(model, z):
            return model.n_dot_ion_obs_vs_tot(z)
        ax_scales = ['linear', 'linear']
        ax_labs = [
            r'$z$', r'Fraction of $\dot n_{\rm ion}$ Probed by UVLF']#r'$\dot n_{\rm ion,obs}(z) / \dot n_{\rm ion} (z)$']  # r'$\langle \dot n_{ion,M>M_{obs}}(z)/\dot n_{ion}(z) \rangle$']
        legend_args = {'fontsize':legend_fontsize}
        def plot_helper(ax):
            ax.set_xticks(np.arange(z_range[0], z_range[1]+1))
            y_label = ax.yaxis.get_label()  # Get the existing x-axis label
            ax.set_ylabel(y_label.get_text(), fontsize=y_size)
        self.plot_vs_z(z_range, func, ax_scales, ax_labs,
                       figsize, ax_lims=ax_lims, hist_plot=hist_plot, kw_func=self.hist_kwargs_func_vs_z, legend_args=legend_args, plot_helper=plot_helper)

    def plot_dn_ion_dz(self, z_range, figsize=[9, 9]):
        def func(model, z): return model.dxi_dz(z, x=0)
        ax_scales = ['linear', 'log']
        ax_labs = [r'$z$', r'$\langle \frac{dn_{ion}}{dz} \rangle$']
        self.plot_vs_z(z_range, func, ax_scales, ax_labs, figsize)

    def plot_big_n_dot_ion(self, z_range, figsize=[10, 5], ax_lims=[None, [50, 52]], legend_fontsize=12):
        def func(model, z): return np.log10(model.big_n_dot_ion(z))
        ax_scales = ['linear', 'linear']
        ax_labs = [
            r'$z$', r'$\log_{10} \dot N_{ion}(z)$ [s$^{-1}$ Mpc$^{-3}$]']

        def plot_helper(ax):
            cs=8
            ax.errorbar(5.1-0.015, 51, yerr=[[0.17], [
                        0.16]], ls='', marker='s', capsize=cs, color='k', label=r'Becker+21 $1\sigma$', markersize=10)
            ax.errorbar(6-0.015, 51.54, yerr=[[0.31], [0.43]],
                        ls='', marker='s', capsize=cs, color='k', markersize=10)
            ax.errorbar(5.1-0.015, 51, yerr=[[0.33], [0.32]], ls='', marker='',
                        capsize=cs, color='k', alpha=0.5, label=r'Becker+21 $2\sigma$', markersize=10)
            ax.errorbar(6-0.015, 51.54, yerr=[[0.62], [1.08]], ls='',
                        marker='', capsize=cs, color='k', alpha=0.5, markersize=10)
            gaikwad_zvals = 0.1 * np.arange(49, 61)
            gaikwad_zvals[2] += 0.015
            gaikwad_zvals[-1] += 0.015
            gaikwad_n_table = np.array([0.6, 0.563, 0.614, 0.668, 0.624, 0.648, 0.587, 0.666, 0.627, 0.609, 0.634, 0.701])
            gaikwad_n = 51 + np.log10(gaikwad_n_table)
            gaikwad_lo_table = np.array([0.248, 0.207, 0.236, 0.238, 0.198, 0.225, 0.149, 0.221, 0.145, 0.141, 0.207, 0.191])
            gaikwad_hi_table = np.array([0.304, 0.352, 0.475, 0.461, 0.421, 0.462, 0.417, 0.402, 0.340, 0.420, 0.343, 0.357])
            gaikwad_lo = np.log10(1 + gaikwad_lo_table/gaikwad_n_table)
            gaikwad_hi = np.log10(1 + gaikwad_hi_table/gaikwad_n_table)
            gaikwad_lo2 = np.log10(1 + 2*gaikwad_lo_table/gaikwad_n_table)
            gaikwad_hi2 = np.log10(1 + 2*gaikwad_hi_table/gaikwad_n_table)
            lab1, lab2 = r'Gaikwad+23 $1\sigma$', r'Gaikwad+23 $2\sigma$' 
            for gz, gn, lo, hi, lo2, hi2 in zip(gaikwad_zvals, gaikwad_n, gaikwad_lo, gaikwad_hi, gaikwad_lo2, gaikwad_hi2):
                ax.errorbar(gz, gn, yerr=[[lo],[hi]], marker='o', capsize=cs, color='red', markersize=10, label=lab1, ls='')
                ax.errorbar(gz, gn, yerr=[[lo2],[hi2]], marker='', capsize=cs, color='red', markersize=10, alpha=0.5, label=lab2, ls='')
                lab1 = lab2 = None
            ax.legend(fontsize=legend_fontsize, ncol=3, loc='upper left')

        self.plot_vs_z(z_range, func, ax_scales, ax_labs,
                       figsize, ax_lims=ax_lims, plot_helper=plot_helper, kw_func=self.kwargs_func_vs_z2)

    def plot_xi(self, z_range, figsize=[10, 5], observed_only=False, legend_fontsize=18, ax_lims=[None, None]):
        def func(model, z):
            return model.xi(z, observed_only=observed_only)
        def func2(model, z):
            return model.xi(z, observed_only=observed_only, f_esc=0.1)
        ax_scales = ['linear', 'linear']
        ax_labs = [r'$z$', r'$\langle x_i(z)\rangle$']
        legend_args = {'fontsize': legend_fontsize, 'ncol':3}

        def plot_helper(ax):
            cs = 5
            lw = 3
            m = utils.MARKER_ARR
            ax.errorbar(8, 1-0.76, yerr=[[0.2], [
                        0]], ls='', marker='', capsize=cs, color='k', label=r'Mason+19', markersize=10, zorder=99, uplims=[True], lw=lw)
            ax.errorbar(7, 1-0.59, yerr=[[0.11], [
                        0.15]], ls='', marker=m[0], capsize=cs, color='red', label=r'Mason+18', markersize=10, zorder=99, lw=lw)
            ax.errorbar(7.09, 1-0.48, yerr=[[0.26], [
                        0.26]], ls='', marker=m[1], capsize=cs, color='blue', label=r'Davies+18', markersize=10, zorder=99, lw=lw)
            ax.errorbar(7.54, 1-0.60, yerr=[[0.20], [
                        0.23]], ls='', marker=m[1], capsize=cs, color='blue', markersize=10, zorder=99, lw=lw)
            ax.errorbar(5.9, 1-0.06, yerr=[[0.06], [
                        0.05]], ls='', marker=m[2], capsize=cs, color='purple', label=r'McGreer+15', markersize=10, zorder=99, lw=lw)
            ax.errorbar(5.6, 1-0.04, yerr=[[0.04], [
                        0.05]], ls='', marker=m[2], capsize=cs, color='purple', markersize=10, zorder=99, lw=lw)
            
            ax.errorbar(7.14, 1-0.46, yerr=[[0.36], [
                        0.32]], xerr=[[0.076], [0.039]], ls='', marker=m[3], capsize=cs, color='C3', markersize=10, zorder=99, lw=lw, label=r'Umeda+23')
            ax.errorbar(7.452, 1-0.54, yerr=[[0.36], [
                        0.32]], xerr=[[0.251], [0.1]], ls='', marker=m[3], capsize=cs, color='C3', markersize=10, zorder=99, lw=lw,)
            ax.errorbar(7.96, 1-0.63, yerr=[[0.26], [
                        0.36]], xerr=[[0.277], [0.586]], ls='', marker=m[3], capsize=cs, color='C3', markersize=10, zorder=99, lw=lw,)
            ax.errorbar(9.801, 1-0.83, yerr=[[0.12], [
                        0.21]], xerr=[[1.164], [1.599]], ls='', marker=m[3], capsize=cs, color='C3', markersize=10, zorder=99, lw=lw,)
            #leg_args = {'handles':ax.get_legend_handles_labels()[0] + [mpl.lines.Line2D([], [], linestyle='-', color='k', label=fr'$f_{{\rm esc}}=0.2$'), mpl.lines.Line2D([], [], linestyle='--', color='k', label=fr'$f_{{\rm esc}}=0.1$')]}
            ax.legend(ncol=2, fontsize=legend_fontsize,)# **leg_args)
        self.plot_vs_z(z_range, [func], ax_scales, ax_labs,
                       figsize, plot_helper=plot_helper, legend_args=legend_args, ax_lims=ax_lims)

    def plot_tau_e(self, z_range, figsize=[10, 5], use_2_sigma=False, legend_fontsize=None, ax_lims=[None, None]):
        def func(model, z):
            return model.tau_e(z)

        def func2(model, z):
            return model.tau_e(z, f_esc=0.1)
        ax_scales = ['linear', 'linear']
        ax_labs = [r'$z$', r'$\tau_e(<z)$']

        def plot_helper(ax):
            legend_args = {} # {'handles':ax.get_legend_handles_labels()[0] + [mpl.lines.Line2D([], [], linestyle='-', color='k', label=fr'$f_{{\rm esc}}=0.2$'), mpl.lines.Line2D([], [], linestyle='--', color='k', label=fr'$f_{{\rm esc}}=0.1$')]}
            legend_1 = ax.legend(loc='lower right', fontsize=legend_fontsize, ncol=2,)# **legend_args)
            planck_best = 0.0561
            planck_bound = 0.0071
            handle = []
            h_label = []
            handle.append(ax.hlines(planck_best, z_range[0], z_range[1], color='k',
                      ls=utils.LINESTYLE_ARR[2], label='Planck 18 Best'))
            h_label.append('Planck+18 Best')
            #ax.hlines(planck_best+planck_bound, z_range[0], z_range[1], color='k',
                      #ls=':', label=r'Planck 18 $\pm1\sigma$')
            #ax.hlines(planck_best-planck_bound, z_range[0], z_range[1], color='k',
                      #ls=':',)
            handle.append(ax.fill_between(z_range, planck_best+planck_bound, planck_best -
                            planck_bound, alpha=0.2, color='k', label=r'Planck+2018 $\pm 1\sigma$'))
            h_label.append(r'Planck+18 $\pm 1\sigma$')
            if use_2_sigma:
                handle.append(ax.fill_between(z_range, planck_best+2*planck_bound, planck_best -
                                2*planck_bound, alpha=0.1, color='k', label=r'Planck+2018 $\pm 2\sigma$'))
                h_label.append(r'Planck+18 $\pm 2\sigma$')
            ax.legend(handle, h_label, loc='upper left', fontsize=legend_fontsize)
            ax.add_artist(legend_1)
            #plt.setp(ax.get_legend().get_texts(), fontsize=legend_fontsize)
        self.plot_vs_z(z_range, [func], ax_scales, ax_labs,
                       figsize, plot_helper=plot_helper, ax_lims=ax_lims, kw_func=self.kwargs_func_vs_z_tau)

    def plot_rho_UV(self, z_range, figsize=[12, 6], legend_fontsize=14, ax_lims=[[5,20], [21.9, 27]], observed=True):
        def func_tot(model, z):
            return np.log10(model.rho_UV(z, observed=False))

        def func_obs(model, z):
            return np.log10(model.rho_UV(z, observed=True))
        ax_scales = ['linear', 'linear']
        ax_labs = [
            r'$z$', r'$\log_{10}(\rho_{\rm UV}$/[erg s$^{-1}$ Hz$^{-1}$ Mpc$^{-3}$])']

        def plot_helper(ax):
            def ax2_func(rho_UV, _=None): return str(rho_UV) + " " + str(_)
    #def ax2_inv_func(rho_SFR): return 10.0**rho_SFR / utils.KAPPA_UV
    #ax2 = ax.secondary_yaxis(
    #   location='right', functions=(ax2_func, ax2_inv_func))
            ax2 = ax.twinx()
            lo, hi = ax.get_ylim()
            print(lo, hi, np.log10(utils.KAPPA_UV))
            ax2.set_ylim(lo+np.log10(utils.KAPPA_UV),
                         hi+np.log10(utils.KAPPA_UV))
            ax2.grid(False)
            ax2.set_ylabel(
                r'$\log_{10}(\rho_{\rm SFR}$/[M$_\odot$ yr$^{-1}$ Mpc$^{-3}$])')

            low_bound = [np.log10(utils.rho_UV_WF(z))[0] for z in z_range]
            upper_min_coupling = [np.log10(utils.rho_UV_WF(z))[
                1] for z in z_range]
            #lower_handle, = ax.plot(z_range, low_bound, color='k', ls=':')
            
            #handles = [lower_handle, minimal_handle]
            #handle_labels = ['WF Coupling Limit', 'Minimal WF Coupling']
            #plt.setp(ax.get_legend().get_texts(), fontsize=legend_fontsize)
            #extra_legend = ax.legend(
            #    handles, handle_labels, loc='upper right', fontsize=legend_fontsize)
            #ax.add_artist(extra_legend)
            legend_1 = ax.legend(fontsize=legend_fontsize, ncol=2)
            minimal_handle = ax.fill_between(
                z_range, low_bound, upper_min_coupling, color='purple', alpha=0.3, label='Minimal WF Coupling')
            print(np.log10(utils.rho_UV_WF(16)))
            ax.plot([16, 16], np.log10(
                utils.rho_UV_WF(16)), color='red', ls='-')
            ax.plot([19, 19], np.log10(
                utils.rho_UV_WF(19)), color='red', ls='-')
            ax.text(16.6, 24.65, 'EDGES', color='red', alpha=1, fontsize=24)
            band_legend = ax.legend([minimal_handle], ['Minimal WF Coupling'], loc='upper right', fontsize=legend_fontsize)
            ax.add_artist(legend_1)


            plt.tight_layout()
        legend_args = {'fontsize': legend_fontsize, 'ncol':1}
        func_arg = [func_tot, func_obs] if observed else func_tot
        self.plot_vs_z(z_range, func_arg, ax_scales, ax_labs,
                       figsize, plot_helper=plot_helper, ax_lims=ax_lims, kw_func=self.rho_kwargs_func_vs_z, legend_args=legend_args, sc_plot=[False, True])
        
    def plot_gamut(self):
        self.plot_fits(n_col=3, legend_fontsize=16, ax_lims=[[None, -12.5], [1e-7,None]])
        self.plot_f_star([6,8], bands=True, tilde=True, legend_args={'fontsize':14, 'loc':'lower center','ncol':2}, band_args={'skip':1003})
        self.plot_LM([6], figsize=[10,5.5])
        self.plot_dn_dot_ion_dlnM([6,8], figsize=[6.5,5], ax_lims=[[1e9, 1e13], [1e-2, 1e1]], legend_args={'fontsize':17.5, 'ncol':1, 'loc':'lower left'})
        self.plot_n_dot_ion_obs_vs_tot([5,10], hist_plot=True, figsize=[6,5], legend_fontsize=14, ax_lims=[None, [0, 1]], y_size=15)
        self.plot_xi([4.5,12], legend_fontsize=14, ax_lims=[[5, 11.5], None])
        self.plot_tau_e([5,15], use_2_sigma=True, legend_fontsize=14, ax_lims=[[5,15], None])
        self.plot_rho_UV([5,20], legend_fontsize=20)

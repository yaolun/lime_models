# this is a small function I wrote to make the typical changes I would do for every plot I make.
def publish(fig, ax=None, figsize=(8,6), cb=None):
    import matplotlib.pyplot as plt
    fig.patch.set_facecolor('white')
    fig.set_figwidth(figsize[0])
    fig.set_figheight(figsize[1])

    if ax == None:
        ax = fig.gca()

    # axis
    ax.minorticks_on()
    [ax.spines[axis].set_linewidth(1.5) for axis in ['top','bottom','left','right']]
    ax.tick_params('both',labelsize=16,width=1.5,which='major',pad=5,length=5)
    ax.tick_params('both',labelsize=16,width=1.5,which='minor',pad=5,length=4)

    # label
    ax.xaxis.label.set_fontsize(18)
    ax.yaxis.label.set_fontsize(18)

    # legend
    if ax.get_legend() != None:
        plt.setp(ax.get_legend().get_texts(), fontsize='16')

    # colorbar
    if cb != None:
        cb.ax.yaxis.label.set_fontsize(18)
        cb.ax.tick_params('both',labelsize=16,width=1.5,which='major',pad=5,length=5)

    return fig, ax

# parameters required by Hyperion2LIME
cs = 0.37
age = 15000
omega = 2.528700e-13
rtout = '/Volumes/SD/model63/model63.rtout'
velfile = '/Volumes/SD/lime_runs/model750/tsc_regrid.h5'
# I store Hyperion2LIME.py at a different directory than the one I am running this notebook,
# so I have to manually add the path to the directory containing Hyperion2LIME.py
import sys
sys.path.append('/Users/yaolun/GoogleDrive/research/lime_models/')

# for auto reload Hyperion2LIME every time it runs (only applicable to the notebook)
%load_ext autoreload
%autoreload 2
#
import Hyperion2LIME as h2l
model = h2l.Hyperion2LIME(rtout, velfile, cs, age, omega, load_full=False)

# this is a quick function to get the 1D abundance
def getLIMEAbundance(configfile, h2l, params=None, external_heating=False, replace_zero=False):
    # load the lime_config
    config_data = ascii.read(configfile)
    config = {}
    for name, val in zip(config_data['col1'],config_data['col2']):
        config[name] = val

    # temperary modify the abundance profile (applied to all models read in)
    if params != None:
        for p in params:
            print(p)
            config[p[0]] = p[1]

    # create r-array based on rMin and rMax
    r_si = 10**np.arange(np.log10(float(config['rMin'])),
                         np.log10(float(config['rMax'])),
                         np.log10(float(config['rMax'])/float(config['rMin']))/1e4)*au_si
    var = np.array([h2l.getAbundance(ir,0,0, config, theta_cav=float(config['theta_cav'])) for i, ir in enumerate(r_si)])

    if replace_zero:
        var[var == 0] = 1e-40

    return (r_si*1e2, var), config


import astropy.constants as const
au = const.au.cgs.value

models = ['model900', 'model901', 'model902', 'model903']

fig = plt.figure()
fig.patch.set_facecolor('white')
ax = fig.add_subplot(111)

# iterate over the given models
for i, mod in enumerate(models):
    # get abundance
    # need to modify the path to the lime_config.txt
    (r, abundance), config = getLIMEAbundance('/Volumes/SD/lime_runs/'+mod+'/lime_config.txt',
                                              model, params=None, external_heating=False, replace_zero=True)
    ax.plot(np.hstack((r[0],r))/au, np.hstack(([1e-40],abundance)), linewidth=3, label=mod)

ax.legend(loc='best', ncol=2)
ax.set_xlabel('Radius [AU]')
ax.set_ylabel('Abundance')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(left=0.5)
ax.set_ylim(bottom=1e-15)
# this is using the function I wrote to make the plot into the style I like
fig, ax = publish(fig)

# You can decide whether you want to write out the figure
# fig.savefig('/Users/yaolun/GoogleDrive/research/bhr71_infall/manuscript/best_abundance_hcop.pdf', format='pdf', dpi=300, bbox_inches='tight')

import matplotlib.pyplot as plt
import pandas as pd

'''
Convenience code to plot the distributions of synsets vs words and words vs. synsets
'''
def matplot_settings():
    from matplotlib.pylab import rcParams
    rcParams['figure.figsize'] = 5,4
    pgf_with_xelatex = {
        'text.usetex': True,
        'text.latex.unicode': True,
        'pgf.rcfonts': False,
        'font.family': 'sans-serif',
        "pgf.texsystem": "xelatex",
        "pgf.preamble": [
            r"\usepackage{amssymb}",
            r"\usepackage{amsmath}",
            r"\usepackage{fontspec}",
            r"\setmainfont{Gentium Book Basic}",
            r"\setsansfont{Open Sans Light}",
            r"\usepackage{unicode-math}",
            r"\setmathfont{TeX Gyre Termes Math}"
        ]
    }
    rcParams.update(pgf_with_xelatex)


def plot(s1, s2):
    plt.clf()
    plt.close('all')
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.plot(s1.index.values, s1.values)
    #ax.set_title('Synsets versus words')
    ax.set_xlabel('Number\_of\_synsets')
    ax.set_ylabel("Number\_of\_words")
    ax = fig.add_subplot(2,1,2)
    ax.plot( s2.index.values, s2.values)
    #ax.set_title('Words versus synsets')
    ax.set_xlabel("Number\_of\_words")
    ax.set_ylabel('Number\_of\_synsets')
    fig=plt.gcf()
    plt.show()
    return fig

if __name__ == '__main__':

    matplot_settings()

    path = 'gold-data/'
    f = '%s%s' % (path, 'synsets-words.csv')
    df = pd.read_csv(f, header=0, sep=',')
    print df.columns
    print len(df.index)
    df.drop(df.columns[[0]], axis=1, inplace=True)
    df.sort_values('N', inplace=True)
    print df[0:3]
    s1 = df.groupby('N')['N'].count()
    print s1
    f = '%s%s' % (path, 'words-synsets.csv')
    df = pd.read_csv(f, header=0, sep=',')
    print df.columns
    print len(df.index)
    df.drop(df.columns[[1]], axis=1, inplace=True)
    df.sort_values('N', inplace=True)
    print df[0:3]
    s2 = df.groupby('N')['N'].count()
    print s2
    fig = plot(s1, s2)
    f = '%s%s' % (path, 'wordssynsets.eps')
    fig.savefig(f, format='eps', dpi=1000)

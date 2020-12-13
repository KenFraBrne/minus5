import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import prince
import sys

from scipy import stats
from sklearn.cluster import KMeans

np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

class task:

    def __init__(self, skiprows=False, printResults=False):
        self.loadData(skiprows)
        self.mkdirPlots()
        if (printResults):
            sys.stdout = open('./results.txt', 'w')

    def loadData(self, skiprows):
        rows = np.arange(1, 1051576, 2) if skiprows else None
        self.df = pd.read_csv('./data.csv', skiprows=rows)

    def mkdirPlots(self):
        self.plotPath = './plots/'
        if ( os.path.exists(self.plotPath) == False ):
            os.mkdir(self.plotPath)

    def plotHist(self):

        n = 20
        xticks = 10**np.arange(0, 6, 1)

        # bet
        # ---
        x = np.logspace(0, 1, n)
        y = self.df.bet_odd
        h, _ = np.histogram(y, bins=x)

        plt.subplot(121)
        plt.semilogx(x[:-1], h, 'k.-')
        plt.xticks(xticks)
        plt.xlim(right=x.max()*1.1)

        # slip
        # ----
        x = np.logspace(0, 6, n)
        y = self.df.slip_odd
        h, _ = np.histogram(y, bins=x)

        plt.subplot(122)
        plt.semilogx(x[:-1], h, 'r.-')
        plt.xlim(x.min(), x.max())

        # show
        # ----
        plt.show()

    def plotSport(self):

        # bet counts by sport
        # -------------------
        count = self.df.groupby('sport').apply(len)
        count = count.sort_values(ascending=False)

        # print percentages to screen
        # ---------------------------
        print('\n   bet fraction    ')
        print('--------------------')
        for s, v in count.iteritems():
            print("%-20s  %6.3f %% ( %d )" % (s, v/count.sum()*100, v))

        # plot
        # ----
        frac = count/count.sum()*100
        y = np.arange(frac.size, 0, -1)
        plt.barh(y, frac)

        plt.grid(True, axis='x')
        plt.title('Fraction of total bets by sport')
        plt.xlabel('Fraction [%]')
        plt.xticks(np.arange(0, 101, 25))
        plt.yticks(y, frac.index.values)
        plt.ylim(y.mean()+2, y.max()+1)
        plt.xlim(0, 100)

        # print & close
        # -------------
        plt.tight_layout()
        plt.savefig(self.plotPath + 'sport.pdf')
        plt.close()

    def plotNoot(self):

        # noot
        # ----
        noot = self.df[[ 'player_id', 'sport' ]].set_index('player_id')
        nootCalc = lambda sport: (sport == 'nogomet').sum()/sport.size*100
        noot = noot.groupby('player_id').apply(nootCalc)
        noot = noot.sort_values('sport').reset_index().sport

        # print noot fraction
        # -------------------
        i = np.argmin(abs(noot-73.01))
        print('\nNumber of noot players: %d [ %d%% ]' % (i, i/noot.size*100))
        print('------------------------------------')

        # plot
        # ----
        plt.plot(noot, np.arange(noot.size), linewidth=2)
        plt.plot([0, noot[i]], np.repeat(i, 2), 'r--')
        plt.plot(np.repeat(noot[i], 2), [0, i], 'r--')

        plt.grid(True, axis='both')
        plt.legend(['$ bet_{nogomet|player} $', '$ bet_{nogomet|total} $', ])
        plt.title("Fraction of each player's 'nogomet' bets")
        plt.yticks(np.arange(0, 3e4, 5e3))
        plt.xticks(np.arange(0, 101, 25))
        plt.ylabel('Player number [-]')
        plt.xlabel('Fraction of nogomet bets [%]')
        plt.ylim(0, noot.size)
        plt.xlim(0, 100)

        # print & close
        # -------------
        plt.tight_layout()
        plt.savefig(self.plotPath + 'noot.pdf')
        plt.close()

    def plotHours(self):

        # hours
        # -----
        hours = self.df[['sport', 'date', 'player_id']].copy()
        hours['hour'] = pd.DatetimeIndex(hours.date).hour
        hours = hours.drop('date', 1).drop_duplicates().drop('player_id', 1)
        hours = hours.set_index('sport')

        # hourHist
        # --------
        f = lambda df: pd.Series( np.histogram( df.hour, np.arange(24) )[0] )
        hourHist = hours.groupby('sport').apply(f)

        i = hourHist.idxmax(1).argsort()
        hourHist = hourHist.iloc[i]

        # print max betting count per sport
        # ---------------------------------
        print('\n    max player hour    ')
        print('------------------------')
        for s, h in hourHist.iterrows():
            c = h.max()
            m = h.argmax()
            f = c/h.sum()*100
            print('%-20s %7d %7.0f %%   %2dh - %2dh' % ( s, c, f, m, m+1 ))

        # plot
        # ----
        y = np.arange(hourHist.shape[0])

        hp = plt.plot(hourHist.idxmax(1), y+1, 'd', markersize=7, color='red')
        for i, s in enumerate( hourHist.index.values ):
            hb = plt.boxplot(hours.loc[s], positions=[i+1], vert=False, widths=0.4, flierprops={ 'markersize' : 2 } )

        plt.title('Players per hour')
        plt.legend( [hp[0], hb['fliers'][0]], ['Max. player/hour', 'outliers'], framealpha=1, loc='upper left')
        plt.grid(True, axis='both', alpha=0.5, linestyle='--')
        plt.xlabel('Time of day [hour]')
        plt.xticks(np.arange(0, 25, 6))
        plt.yticks(y+1, hourHist.index.values)
        plt.xlim(-0.5, 24)

        # print
        # -----
        plt.tight_layout()
        plt.savefig(self.plotPath + 'hours.pdf')
        plt.close()

    def plotLive(self, nboot=100):

        # bets
        # ----
        bets = self.df[['bet_type', 'sport', 'bet_odd']].copy()
        bets = bets.set_index([ 'bet_type', 'sport' ])
        bets = bets.sort_index()
        bets['log10'] = np.log10(bets.bet_odd)

        # stat tests
        # ----------
        def fun(df):
            l = df.index.unique()
            d = df.bet_odd
            [ ks, ks_p, wx, wx_p ] = [ np.nan ] * 4
            if (l.size > 1):
                o = d.loc['live']
                m = d.loc['prematch']
                ks, ks_p = stats.ks_2samp(o, m, alternative='greater')
                wx, wx_p = stats.ranksums(o, m)
            return pd.DataFrame({
                'ks' : ks,
                'wx' : wx,
                'ks_p' : ks_p,
                'wx_p' : wx_p,
            }, index=[0])

        tests = bets.groupby('sport').apply(fun).droplevel(1)

        print('\n' + ' '*22 + 'test results')
        print('-'*62)
        print(' '*32 + '%-18s' % 'ks test' + 'wx test')
        for s, r in tests.iterrows():
            print('%-30s %5.2f ( %4.2f )    %5.2f ( %4.2f )' % (s, r.ks, r.ks_p, r.wx, r.wx_p))

        # bootstrap log10 means (with loops due to memory)
        # ------------------------------------------------
        def fun(df):
            mu = np.zeros(nboot)
            for i in range(nboot):
                mu[i] = np.random.choice( df.log10, df.log10.size ).mean()
            mu = np.sort(mu)
            return pd.Series(mu)

        boot = bets.groupby(['bet_type', 'sport']).apply(fun)

        print('\n     bootstrap means ( %d samples )     ' % nboot)
        print('---------------------------------------------')
        iq = ( np.array([0.5, 0.025, 0.975])*nboot ).astype('int')
        for (l, s), b in boot.iterrows():
            out = (s, l,) + tuple(b[iq])
            print(' %-20s %-10s %5.3f (%5.3f-%5.3f)' % out)

        # reorder categories by bootstrap mean
        # ----------------------
        cnew = boot.iloc[:, iq[0]]
        cnew = cnew.sort_values().reset_index()
        cnew = cnew.drop_duplicates('sport').reset_index()

        # plot bootstraped means
        # ----------------------
        colorFun = lambda bet_type : 'C1' if bet_type == 'live' else 'C0'

        for (l, s), v in boot.iterrows():
            colorDict = { 'color' : colorFun(l) }
            props = {
                'vert' : False,
                'widths' : 0.3,
                'patch_artist': True,
                'boxprops' : { **colorDict, 'facecolor' : colorFun(l) },
                'capprops' : { **colorDict },
                'medianprops' : { **colorDict },
                'whiskerprops' : { **colorDict },
                'flierprops' : {
                    'marker' : '.',
                    'markersize' : 3,
                    'markerfacecolor' : colorFun(l),
                    'markeredgecolor' : colorFun(l),
                },
            }
            y = cnew[cnew.sport == s].index.values
            plt.boxplot(v, positions=y, **props)

        plt.xlabel('$log_{10}$(bet_odd)')
        plt.xticks(np.arange(0, 0.81, 0.2))
        plt.yticks(np.arange(cnew.sport.size), cnew.sport)
        plt.title('bootstrapped mean ( %d samples )' % nboot)
        plt.grid(True, axis='both', alpha=0.5, linestyle='--')

        hl = plt.gca().lines
        plt.legend([hl[0], hl[-3]], ['live', 'prematch'])

        plt.tight_layout()
        plt.savefig(self.plotPath + 'live.pdf')
        plt.clf()

        # plot distributions
        # ------------------
        bins = np.linspace(0, 1, 30)
        for s, df in bets.groupby('sport'):
            df = df.droplevel('sport').log10

            # prepare
            # -------
            fig = plt.figure(figsize=(8, 4))
            groups = df.index.unique()
            hist = [ np.histogram( df.loc[g], bins )[0]/df.loc[g].size for g in groups ]
            xcdf = [ np.sort(df.loc[g]) for g in groups ]
            ycdf = [ np.linspace(0, 1, x.size) for x in xcdf ]

            # plot hist
            # ---------
            plt.subplot(121)
            for i, h in enumerate(hist):
                plt.step(bins[1:], h, where='post', color=colorFun(groups[i]))
            plt.legend(groups)
            plt.title("%s" % s)
            plt.xlabel('$log_{10}$(bet_odd)')
            plt.ylabel('Relative frequency')
            plt.xlim(0, 1)
            plt.ylim(0, 0.4)
            plt.xticks(np.arange(0, 1.1, .25))
            plt.yticks(np.arange(0, 0.41, 0.1))

            # plot cdf
            # --------
            plt.subplot(122)
            for i, (x, y) in enumerate(zip(xcdf, ycdf)):
                plt.plot(x, y, color=colorFun(groups[i]))
            plt.title("%s" % s)
            plt.xlabel('$log_{10}$(bet_odd)')
            plt.ylabel('CDF')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xticks(np.arange(0, 1.1, .25))
            plt.yticks(np.arange(0, 1.1, .25))

            # print
            # -----
            plt.tight_layout()
            fig.savefig(self.plotPath + 'live_%s.pdf' % s.replace(' ', '_'))
            plt.clf()

    def plotGroups(self):

        # df
        # --
        cols = ['player_id', 'bet_type', 'sport', 'bet_odd', 'slip_odd']
        df = self.df[cols].copy()

        # boxcox & scale odds
        # -------------------
        df.bet_odd, _ = stats.boxcox( df.bet_odd )
        df.slip_odd, _ = stats.boxcox( df.slip_odd )

        # calculate ( player_id x bet_type ) means
        # ----------------------------------------
        X = df.groupby(['player_id', 'bet_type']).mean()
        X = X.reset_index('bet_type')

        # Shannon's index
        # ---------------
        def calcH(df):
            _, n = np.unique( df.codes, return_counts=True )
            p = n/n.sum()
            return -np.sum(p*np.log(p))
        H = df[['player_id', 'sport']].copy().set_index('player_id')
        H.sport = pd.Categorical(H.sport)
        H['codes'] = H.sport.cat.codes
        H = H.groupby('player_id').apply(calcH)
        H = (H-H.min())/np.ptp(H)
        X['H_index'] = X.index.to_series().map(H.to_dict())

        # scale floats between 0 and 1
        # ----------------------------
        for c, v in X.iteritems():
            if ( v.dtype == 'float64' ):
                X[c] = (v-v.min())/np.ptp(v)

        # prince & kmeans
        # ---------------
        famd = prince.FAMD(n_components=4, random_state=42)
        famd = famd.fit(X)

        inrt = famd.explained_inertia_*100
        corr = famd.column_correlations(X)

        Xrc = famd.row_coordinates(X)
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans = kmeans.fit(Xrc)

        print('\n           corr')
        print('-----------------------------------')
        for t, v in corr.iterrows():
            print('%-20s' % t + ' %5.2f'*v.size % tuple(v))

        print('\n           corr**2')
        print('-----------------------------------')
        for t, v in corr.iterrows():
            print('%-20s' % t + ' %5.2f'*v.size % tuple(v**2))

        print('\n           inertia')
        print('-----------------------------------')
        for iv in enumerate(inrt):
            print('Component %d  %5.2f' % iv)

        # plot famd
        # ---------
        plt.figure(figsize=(8, 4))

        plt.subplot(121)
        for t, r in corr.iterrows():
            plt.plot([0, r[0]], [0, r[2]], 'k', linewidth=0.7)
            plt.plot(r[0], r[2], 'ko', markersize=4)
            plt.text(r[0], r[2], t)

        plt.title('Variable correlation')
        plt.xlabel('Component 0 ( %0.2f %% )' % (inrt[0]))
        plt.ylabel('Component 2 ( %0.2f %% )' % (inrt[2]))
        plt.grid(True, axis='both', alpha=0.5, linestyle='--')
        plt.xticks(np.arange(-1, 1.1, .5))
        plt.yticks(np.arange(-1, 1.1, .5))

        # plot inertia
        # ------------
        plt.subplot(122)
        plt.bar(np.arange(inrt.size), inrt)

        plt.xticks(np.arange(inrt.size))
        plt.title('Inertia explained')
        plt.xlabel('Component')
        plt.ylabel('%')

        # print famd & inertia
        # --------------------
        plt.tight_layout()
        plt.savefig(self.plotPath + 'famd.pdf')
        plt.close()

        # plot kmeans
        # -----------
        plt.figure(figsize=(8, 4))

        l = kmeans.labels_
        rc = famd.row_coordinates(X)
        x = rc[0]
        y = rc[1]

        n = 30
        xb = np.linspace(-3, 3, n-1)
        yb = np.linspace(-4, 4, n)
        zb = [ np.histogram2d( x[l==i], y[l==i], bins=(xb, yb) )[0] for i in range(3) ]
        xb, yb = np.meshgrid( (xb[:-1]+xb[1:])/2, (yb[:-1]+yb[1:])/2 )

        nlev = 8
        colors = [ plt.cm.Blues, plt.cm.Oranges, plt.cm.Greens ]

        plt.subplot(121)
        for z, c in zip(zb, colors):
            plt.contour(
                xb, yb, z.T,
                levels=np.linspace(2, z.max(), nlev),
                colors=c(np.linspace(0, 1, nlev)),
            )

        for i, (x, y) in enumerate(kmeans.cluster_centers_[:, :2]):
            plt.plot( x, y, 'd', markersize=7)

        plt.grid(True, linestyle='--', alpha=0.5)
        plt.title('K-means clusters')
        plt.xlabel('Component 0')
        plt.ylabel('Component 1')
        plt.xlim(-2, 3)
        plt.ylim(-2, 3)

        # plot bets by category
        # ---------------------
        nh = 30
        xh = np.linspace(0, 1, nh)

        plt.subplot(122)
        for i in range(3):
            yh, _ = np.histogram( X.bet_odd[ l == i ], xh, density=True )
            plt.step(xh[1:], yh)

        plt.xticks(np.arange(0, 1.1, 1/4))
        plt.yticks(np.arange(0, 4.1))
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.title('bet_odd PDF')
        plt.xlabel('bet_odd ( boxcox & scaled )')
        plt.ylabel('PDF')
        plt.xlim(0, 1)
        plt.ylim(0, 4)

        # print kmeans & labeled bets
        # ---------------------------
        plt.tight_layout()
        plt.savefig(self.plotPath + 'kmeans.pdf')
        plt.close()

def main():

    plt.clf()
    nboot = 10000
    skiprows = False
    printResults = True

    t = task(skiprows=skiprows, printResults=printResults)
    t.plotSport()
    t.plotNoot()
    t.plotHours()
    t.plotLive(nboot=nboot)
    t.plotGroups()

    return t

if __name__ == '__main__':
    t = main()

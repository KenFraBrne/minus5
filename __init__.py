from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import prince
import sys

np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)

class task:

    def __init__(self, skiprows=False):
        self.loadData(skiprows)
        self.mkdirPlots()

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

        # get unique sport counts
        # -----------------------
        sport, count = np.unique(self.df.sport, return_counts=True)
        frac = count/count.sum()*100
        ind = np.argsort(frac)

        # print percentages to screen
        # ---------------------------
        print('\n   sport percentages    ')
        print('------------------------')
        for i in ind[::-1]:
            print("%-20s  %5.2f %%" % (sport[i], frac[i]))

        # plot
        # ----
        y = np.arange(sport.size)
        plt.barh(y, frac[ind])

        plt.grid(True, axis='x')
        plt.title('Fraction of total bets by sport')
        plt.xlabel('Fraction [%]')
        plt.xticks(np.arange(0, 101, 25))
        plt.yticks(y, sport[ind])
        plt.ylim(y.mean()+2, y.max()+1)
        plt.xlim(0, 100)

        # print & close
        # -------------
        plt.tight_layout()
        plt.savefig(self.plotPath + 'sport.png')
        plt.close()

    def plotNoot(self):

        # prepare
        # -------
        df = self.df
        player_id = df.player_id
        foot = df.sport == 'nogomet'
        array = np.vstack((player_id, foot)).T

        # non foot bet count (where applicable)
        # -------------------------------------
        nootPlayer, nootCount = np.unique(array, axis=0, return_counts=True)
        i = nootPlayer[:, 1] != 1
        nootPlayer = nootPlayer[i, 0]
        nootCount = nootCount[i]

        # total bet count (where applicable)
        # ----------------------------------
        player, betCount = np.unique(player_id, return_counts=True)
        i = np.in1d(player, nootPlayer)
        player = player[i]
        betCount = betCount[i]

        # non foot fractions
        # ------------------
        nPlayer = player_id.unique().size
        frac = np.hstack((np.zeros(nPlayer - nootPlayer.size), nootCount/betCount))
        frac = 100 - np.sort(frac)*100
        frac = frac[::-1]

        # print count & fraction of noot players
        # --------------------------------------
        i = np.argmin(abs(73 - frac))
        print('\nNumber of noot players: %d [ %d%% ]' % (i, i/nPlayer*100))
        print('------------------------------------')

        # plot
        # ----
        plt.plot(frac, np.arange(nPlayer), linewidth=2)
        plt.plot([0, frac[i]], np.repeat(i, 2), 'r--')
        plt.plot(np.repeat(frac[i], 2), [0, i], 'r--')

        plt.grid(True, axis='both')
        plt.legend(['$ bet_{nogomet|player} $', '$ bet_{nogomet|total} $', ])
        plt.title("Fraction of 'nogomet' bets")
        plt.yticks(np.arange(0, 3e4, 5e3))
        plt.xticks(np.arange(0, 101, 25))
        plt.ylabel('Player number [-]')
        plt.xlabel('Fraction [%]')
        plt.ylim(0, nPlayer)
        plt.xlim(0, 100)

        # print & close
        # -------------
        plt.tight_layout()
        plt.savefig(self.plotPath + 'noot.png')
        plt.close()

    def plotHourBets(self):

        # prepare
        # -------
        sports = np.unique(self.df.sport)
        date = pd.DatetimeIndex(self.df.date)
        hour = date.hour

        # get hour data per sport
        # -----------------------
        sportHours = np.array([hour[self.df.sport == sport]
                               for sport in sports])
        bins = np.arange(24)
        hourHists = np.array([np.histogram(hours, bins)
                              for hours in sportHours])
        hourCounts, _ = zip(*hourHists)
        hourCounts = np.array(hourCounts)
        hourMax = np.argmax(hourCounts, axis=1)

        # print mean betting hour per sport
        # ---------------------------------
        for i, sport in enumerate(sports):
            print('Max betting hour for %-20s: %d' % (sport, hourMax[i]))

        # plot
        # ----
        i = np.argsort(hourMax)
        y = np.arange(sports.size)+1

        plt.boxplot(sportHours[i], sym='', vert=False)
        h = plt.plot(hourMax[i], y, 'ro')

        plt.legend( h, ['Max. bet hour'], fontsize='x-large', framealpha=1 )
        plt.title('Bets per hour')
        plt.xlabel('Time of day [hour]')
        plt.xlim(0, 24)
        plt.xticks(np.arange(0, 25, 6))
        plt.yticks(y, sports[i])

        # print
        # -----
        plt.tight_layout()
        plt.savefig(self.plotPath + 'hourBets.png')
        plt.close()

    def plotHourPlayers(self):

        # prepare
        # -------
        sport = self.df.sport
        sports, sportInd = np.unique(sport, return_inverse=True)

        player_id = self.df.player_id
        nPlayer = np.unique(player_id).size

        date = pd.DatetimeIndex(self.df.date)
        hour = date.hour

        # unique (sport, hour, player) combinations to remove multiple bets
        # -----------------------------------------------------------------
        array = np.vstack((sportInd, hour, player_id)).T
        shp = np.unique(array, axis=0)
        s = shp[:,0]
        h = shp[:,1]

        # sportHours
        # ----------
        bins = np.arange(24)
        sportHours = np.array( [ h[ s == sVal ] for sVal in np.unique(s) ], dtype=object )
        ret = [ np.histogram( sHours, bins ) for sHours in sportHours ]
        playerCounts, _ = zip(*ret)
        playerCounts = np.array(playerCounts)
        playerMax = np.max(playerCounts, axis=1)
        playerArgmax = np.argmax(playerCounts, axis=1)
        ind = np.argsort(playerMax)

        # print max betting count per sport
        # ---------------------------------
        print('\n    max player number    ')
        print('-------------------------')
        for i in ind[::-1]:
            print('%-20s  %5.2f %%  %2d h' % (sports[i], playerMax[i]/nPlayer*100, playerArgmax[i]))

        # plot
        # ----
        i = np.argsort(playerArgmax)
        y = np.arange(sports.size)

        hp = plt.plot(playerArgmax[i], y+1, 'd', markersize=7, color='red')
        hb = plt.boxplot(sportHours[i], vert=False, widths=0.4, flierprops={ 'markersize' : 2 })

        plt.title('Players per hour')
        plt.legend( [hp[0], hb['fliers'][0]], ['Max. player/hour', 'outliers'], framealpha=1, loc='upper left')
        plt.grid(True, axis='both', alpha=0.5, linestyle='--')
        plt.xlabel('Time of day [hour]')
        plt.xticks(np.arange(0, 25, 6))
        plt.yticks(y+1, sports[i])
        # plt.xlim(-0.5, 24)

        # print
        # -----
        plt.tight_layout()
        plt.savefig(self.plotPath + 'hourPlayers.png')
        plt.close()

    def plotLive(self, nBoot=100):

        # prepare
        # -------
        bets = self.df.bet_odd
        live = self.df.bet_type == 'live'
        sport = self.df.sport
        sports, sportInd = np.unique( sport, return_inverse=True )

        X = np.vstack((live, sportInd)).T
        X, Xinv = np.unique(X, axis=0, return_inverse=True)
        Y = [ bets[Xinv == inv] for inv in np.unique(Xinv) ]
        Ylog10 = np.array([ np.log10(y) for y in Y ], dtype=object)

        # bootstrap means (with loops because of memory)
        # ----------------------------------------------
        row = np.zeros(nBoot)
        Yboot = np.zeros((Ylog10.size, nBoot))
        iq = (nBoot*np.array([5e-1, 5e-2, 95e-2])).astype('int')

        print('\n     bootstrap means ( %d samples )     ' % nBoot)
        print('---------------------------------------------')
        for i in range(Yboot.shape[0]):
            for j in range(Yboot.shape[1]):
                row[j] = np.random.choice( Ylog10[i], Ylog10[i].size ).mean()
            Yboot[i] = np.sort(row)

            # print
            # -----
            output = (i, X[i, 1], sports[X[i, 1]],)
            output += ('live' if X[i, 0] else 'prematch',)
            output += tuple(Yboot[i][iq])
            print('%2d  %2d  %-20s %10s   %5.3f   ( %5.3f - %5.3f )' % output)

        # histograms
        # ----------
        nbin = 30
        bins = np.arange(0, 1+2/nbin, 1/nbin)

        Yhist = np.array([ np.histogram(y, bins)[0] for y in Ylog10 ], dtype=object)
        Yhist = Yhist/Yhist.sum(1, keepdims=True)

        # test 'live' differences
        # -----------------------
        print('\n     ks test results     ')
        print('------------------------------------')

        ksResult = np.full((sports.size, 2), np.nan)
        for iSport in range(sports.size):

            # calc
            # ----
            i = X[:,1] == iSport
            y = [ a for a, b in zip(Ylog10, i) if b ]
            if (len(y) > 1):
                yo = y[0]
                ym = y[1]
                score, pvalue = stats.ks_2samp(ym, yo, 'greater')
                ksResult[iSport] = [score, pvalue]

            # print
            # -----
            output = (sports[iSport],) + tuple(ksResult[iSport])
            print('%-20s %5.2f (%5.2f)' % output)

        # remap old sport categories (sold) based on the mean
        # ---------------------------------------------------
        sold = X[:, 1]
        mean = Yboot[:, nBoot//2]
        _, i = np.unique( sold, return_index=True )
        j = mean[i].argsort()
        cnew = sold[i][j].tolist()
        X[:, 1] = np.array([ cnew.index(cold) for cold in sold ])
        ksResult = ksResult[cnew]
        sports = sports[cnew]

        # plot bootstraped means
        # ----------------------
        xl = X[:, 0]
        xs = X[:, 1]
        ys = Yboot[:]

        _, indTick = np.unique( xs, return_index=True )
        xt = xs[indTick]
        tt = sports[xt]

        for i, (x, y) in enumerate(zip(xs, ys)):
            color = 'C1' if xl[i] else 'C0'
            colorDict = { 'color' : color }
            props = {
                'vert' : False,
                'widths' : 0.25,
                'boxprops' : { **colorDict },
                'capprops' : { **colorDict },
                'medianprops' : { **colorDict },
                'whiskerprops' : { **colorDict },
                'flierprops' : { 'marker' : '.', 'markersize' : 3, 'markerfacecolor' : color, 'markeredgecolor' : color, },
            }
            plt.boxplot(y, positions=[x], **props)

        plt.yticks(xt, tt)
        plt.xlabel('$log_{10}$(bet_odd)')
        plt.xticks(np.arange(0, 0.81, 0.2))
        plt.title('bootstrapped mean ( %d samples )' % nBoot)
        plt.grid(True, axis='both', alpha=0.5, linestyle='--')

        hl = plt.gca().lines
        plt.legend([hl[0], hl[-3]], ['prematch', 'live'])

        plt.tight_layout()
        plt.savefig(self.plotPath + 'live.png')
        plt.clf()

        # plot distributions
        # ------------------
        for iSport in range(sports.size):

            # prepare
            # -------
            fig = plt.figure(figsize=(8, 4))
            i = X[:,1] == iSport
            isLive = X[i][:,0]

            # plot hist
            # ---------
            plt.subplot(121)

            hp = plt.step(bins[:-1], Yhist[i].T)

            if ( isLive.size > 1 ):
                plt.legend(['prematch', 'live'])
            else:
                if ( isLive == 1 ):
                    hp[0].set_color('C1')
                    plt.legend([ 'live' ])
                else:
                    hp[0].set_color('C0')
                    plt.legend([ 'prematch' ])

            plt.title("%s" % sports[iSport])
            plt.xlabel('$log_{10}$(bet_odd)')
            plt.ylabel('rel. frequency')
            plt.xlim(0, 1)
            plt.ylim(0, 0.4)
            plt.xticks(np.arange(0, 1.1, .25))
            plt.yticks(np.arange(0, 0.41, 0.1))

            # plot cdf
            # --------
            plt.subplot(122)
            xp = [ np.sort(y) for y in Ylog10[i] ]
            yp = [ np.linspace(0, 1, x.size) for x in xp ]
            for x, y in zip(xp, yp):
                hp = plt.plot(x, y)

            if ( isLive[0] == 1 ):
                hp[0].set_color('C1')

            plt.title("%s" % sports[iSport])
            plt.xlabel('$log_{10}$(bet_odd)')
            plt.ylabel('CDF')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.xticks(np.arange(0, 1.1, .25))
            plt.yticks(np.arange(0, 1.1, .25))

            # print
            # -----
            plt.tight_layout()
            fig.savefig(self.plotPath + 'live_%s.png' % sports[iSport].replace(' ', '_'))
            plt.clf()

    def plotGroups(self):

        # df
        # --
        columns = [
            'bet_odd',
            'slip_odd',
            'bet_type',
        ]
        data = self.df[columns]
        index = self.df.player_id
        X = pd.DataFrame(data=data, index=index, columns=columns)

        # insert hour
        # -----------
        date = pd.DatetimeIndex(self.df.date)
        X['hour'] = date.hour

        # boxcox odds
        # -----------
        X.bet_odd, betLambda = stats.boxcox(X.bet_odd)
        X.slip_odd, slipLambda = stats.boxcox(X.slip_odd)

        # zscore
        # ------
        for col in X.columns:
            if ( X[col].dtype != 'object' ):
                X[col] = stats.zscore( X[col] )

        # prince
        # ------
        famd = prince.FAMD(
            n_components=X.columns.size,
            n_iter=3,
            copy=True,
            check_input=True,
            engine='auto',
        )
        famd = famd.fit(X)

        # group
        # -----

        # plot correlation & inertia contribution
        # ---------------------------------------
        fig = plt.figure(figsize=(8, 4))

        r2 = famd.column_correlations(X)
        inert = famd.explained_inertia_*100

        plt.subplot(121)
        u = r2[0]
        v = r2[1]
        plt.quiver(u*0, v*0, u, v, angles='xy', scale_units='xy', scale=1, width=3e-3, headwidth=7)
        for i, t in enumerate(r2.index):
            plt.text(u[i], v[i], t)

        plt.xlim(-1, 1)
        plt.ylim(-1, 1)
        plt.xticks(np.arange(-1, 1.1, 0.5))
        plt.yticks(np.arange(-1, 1.1, 0.5))
        plt.grid(True, axis='both', alpha=0.5, linestyle='--')
        plt.xlabel('Component 0 ( %0.2f %% )' % (inert[0]))
        plt.ylabel('Component 1 ( %0.2f %% )' % (inert[1]))
        plt.title('Variable correlation')

        plt.subplot(122)
        plt.bar(np.arange(inert.size), inert)

        plt.xticks(np.arange(inert.size))
        plt.title('Inertia explained')
        plt.xlabel('Component')
        plt.ylabel('%')

        plt.tight_layout()
        plt.savefig('./famd.png')
        plt.close()

        # return
        # ------
        return famd, X

def main():

    plt.clf()
    nBoot = 100
    skiprows = True

    t = task(skiprows=skiprows)
    # t.plotSport()
    # t.plotNoot()
    # t.plotHourPlayers()
    # t.plotLive(nBoot=nBoot)
    famd, X = t.plotGroups()

    return t, famd, X

if __name__ == '__main__':
    t, famd, X = main()

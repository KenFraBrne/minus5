from scipy import stats
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

class task:

    def __init__(self):
        self.loadData()
        self.mkdirLivePlots()

    def loadData(self):
        rows = np.arange(1, 1051576, 2)
        # rows = None
        self.df = pd.read_csv('./data.csv', skiprows=rows)

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
        plt.xlim(0, 100)
        plt.xticks(np.arange(0, 101, 25))
        plt.yticks(y, sport[ind])
        plt.ylim(y.mean(), y.max()+1)

        # print & close
        # -------------
        plt.tight_layout()
        plt.savefig('./sport.png')
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
        plt.title("Fraction of 'nogomet' bets ECDF")
        plt.xlabel('Fraction [%]')
        plt.xlim(0, 100)
        plt.xticks(np.arange(0, 101, 25))
        plt.ylabel('Player number [-]')
        plt.ylim(0, nPlayer)
        plt.yticks(np.arange(0, 3e4, 5e3))

        # print & close
        # -------------
        plt.savefig('./noot.png')
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
        plt.savefig('./hourBets.png')
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
        sportHours = [ h[ s == sVal ] for sVal in np.unique(s) ]
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

        plt.boxplot(np.array(sportHours)[i], sym='', vert=False)
        h = plt.plot(playerArgmax[i]+0.5, y+1, 'ro')

        plt.legend( h, ['Max. player/hour'], fontsize='large', framealpha=1)
        plt.title('Players per hour')
        plt.xlabel('Time of day [hour]')
        plt.xlim(0, 24)
        plt.xticks(np.arange(0, 25, 6))
        plt.yticks(y+1, sports[i])

        # print
        # -----
        plt.tight_layout()
        plt.savefig('./hourPlayers.png')

    def mkdirLivePlots(self):
        self.livePlotsPath = './livePlots/'
        if ( os.path.exists(self.livePlotsPath) == False ):
            os.mkdir(self.livePlotsPath)

    def plotLive(self):

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
        nBoot = 100
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
            print('%2d %2d %-20s %10s   %5.3f   ( %5.3f - %5.3f )' % output)

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

        # remap sport categories according to means
        # -----------------------------------------
        l = X[:, 0]
        s = X[:, 1]
        m = Yboot[:, nBoot//2]
        _, i = np.unique( X[:, 1], return_index=True )
        snew = s[i][ m[i].argsort() ]
        Xnew = np.array([ snew[sold] for sold in s ])

        # plot bootstraped means
        # ----------------------
        xl = X[:, 0]
        xs = X[:, 1]
        ys = Yboot

        _, indTick = np.unique( xs, return_index=True )
        tt = sports[xs[indTick]]
        xt = xs[indTick]

        for i, (x, y) in enumerate(zip(xs, ys)):
            color = 'C1' if xl[i] else 'C0'
            colorDict = { 'color' : color }
            props = {
                'vert' : False,
                'widths' : 0.4,
                'boxprops' : { **colorDict },
                'capprops' : { **colorDict },
                'medianprops' : { **colorDict },
                'whiskerprops' : { **colorDict },
                'flierprops' : { 'marker' : '.', 'markersize' : 5, 'markerfacecolor' : color, 'markeredgecolor' : color, },
            }
            plt.boxplot(y, positions=[x], **props)
            plt.plot(y[nBoot//2], x, 'ro')

        plt.yticks(xt, tt)
        plt.grid(True, axis='y', alpha=0.4, linestyle='--')
        plt.tight_layout()
        plt.savefig('./plotLive.png')
        plt.clf()

        return

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
            fig.savefig(self.livePlotsPath + '%s.png' % sports[iSport].replace(' ', '_'))
            plt.clf()

def main():
    plt.clf()
    t = task()
    # t.plotSport()
    # t.plotNoot()
    # t.plotHourPlayers()
    t.plotLive()
    return t

if __name__ == '__main__':
    np.warnings.filterwarnings('error', category=np.VisibleDeprecationWarning)
    t = main()

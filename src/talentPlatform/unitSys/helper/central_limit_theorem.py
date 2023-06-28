import random
from src.talentPlatform.unitSys.helper.InitConfig import *

class CentralLimitTheorem(object):
    # ================================================================================================
    # 초기 환경변수 설정
    # ================================================================================================
    global log

    log = initLog(env = None, contextPath = None, prjName = None)

    def __init__(self, distribution):
        self.distribution = distribution
        self.dist_min = np.min(distribution)
        self.dist_max = np.max(distribution)
        self.dist_mean = np.mean(distribution)

    def _sample(self, N):
        sampleSum = 0
        for i in range(N):
            sampleSum += random.choice(self.distribution)
        return float(sampleSum) / float(N)

    def run_sample(self, N, plot=False, num_bins=None):
        means = []
        for i in range(10000):
            means.append(self._sample(N))
        if plot:
            title = "Sample Mean Distribution with N = %s" % N
            # plot_distribution(means, title, self.dist_min, self.dist_max, num_bins)
            log.info("[샘플 크기: %s개] 모집단 평균 - 표본 평균 : {%s}", N, round(self.dist_mean - np.mean(means), 2))

        return means


def plot_distribution(distribution, title=None, bin_min=None, bin_max=None, num_bins=None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_palette("deep", desat=.65)
    if num_bins != None:
        bin_size = (bin_max - bin_min) / num_bins
        manual_bins = range(bin_min, bin_max + bin_size, bin_size)
        [n, bins, patches] = plt.hist(distribution, bins=manual_bins)
    else:
        [n, bins, patches] = plt.hist(distribution)
    if title != None:
        plt.title(title)
    plt.xlim(bin_min, bin_max)
    plt.ylim(0, max(n) + 2)
    plt.ylabel("Frequency")
    plt.xlabel("Observation")
    plt.show()

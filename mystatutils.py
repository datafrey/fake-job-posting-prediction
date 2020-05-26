'''For quick access to statistical methods.'''
import numpy as np
from scipy import stats
import itertools


# -------------------------- Confidence intervals --------------------------

def zconfint(sample, sigma=None, alpha=0.05):
    '''Confidence interval based on normal distribution.'''
    mean = np.mean(sample)
    n = len(sample)

    if not sigma:
        sigma = np.std(sample)

    z = stats.norm.ppf(1 - alpha / 2)
    left_boundary = mean - z * sigma / np.sqrt(n)
    right_boundary = mean + z * sigma / np.sqrt(n)

    return left_boundary, right_boundary


def zconfint_diff(sample1, sample2, sigma1=None, sigma2=None, alpha=0.05):
    '''Confidence interval based on normal distribution for
    the difference in means of two samples.'''
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    n1 = len(sample1)
    n2 = len(sample2)

    if not sigma1:
        sigma1 = np.std(sample1)

    if not sigma2:
        sigma2 = np.std(sample2)

    z = stats.norm.ppf(1 - alpha / 2)
    left_boundary = (mean1 - mean2) - z * np.sqrt((sigma1 ** 2) / n1 + (sigma2 ** 2) / n2)
    right_boundary = (mean1 - mean2) + z * np.sqrt((sigma1 ** 2) / n1 + (sigma2 ** 2) / n2)

    return left_boundary, right_boundary


def tconfint(sample, alpha=0.05):
    '''Confidence interval based on Student t distribution.'''
    mean = np.mean(sample)
    S = np.std(sample, ddof=1)
    n = len(sample)

    t = stats.t.ppf(1 - alpha / 2, n - 1)
    left_boundary = mean - t * S / np.sqrt(n)
    right_boundary = mean + t * S / np.sqrt(n)

    return left_boundary, right_boundary


def tconfint_diff(sample1, sample2, alpha=0.05):
    '''Confidence interval based on Student t distribution for
    the difference in means of two samples.'''
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    s1 = np.std(sample1, ddof=1)
    s2 = np.std(sample2, ddof=1)
    n1 = len(sample1)
    n2 = len(sample2)

    sem1 = np.var(sample1) / (n1 - 1)
    sem2 = np.var(sample2) / (n2 - 1)
    semsum = sem1 + sem2
    z1 = (sem1 / semsum) ** 2 / (n1 - 1)
    z2 = (sem2 / semsum) ** 2 / (n2 - 1)
    dof = 1 / (z1 + z2)

    t = stats.t.ppf(1 - alpha / 2, dof)
    left_boundary = (mean1 - mean2) - t * np.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)
    right_boundary = (mean1 - mean2) + t * np.sqrt((s1 ** 2) / n1 + (s2 ** 2) / n2)

    return left_boundary, right_boundary


def bootstrap_confint(sample, stat=np.mean, n_samples=5000, alpha=0.05):
    '''Confidence interval for a `stat` of a `sample` calculation
    using bootstrap sampling mechanism. `stat` is a numpy function
    like np.mean, np.std, np.median, np.max, np.min, etc.'''
    indices = np.random.randint(0, len(sample), (n_samples, len(sample)))
    samples = sample[indices]

    stat_scores = stat(samples, axis=1)
    boundaries = np.percentile(stat_scores, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return boundaries


def bootstrap_confint_diff(sample1, sample2, stat=np.mean, n_samples=5000, alpha=0.05):
    '''Confidence interval for a difference in `stat` of two samples
    calculation using bootstrap sampling mechanism. `stat` is a numpy
    function like np.mean, np.std, np.median, np.max, np.min, etc.'''
    indices1 = np.random.randint(0, len(sample1), (n_samples, len(sample1)))
    indices2 = np.random.randint(0, len(sample2), (n_samples, len(sample2)))
    samples1 = sample1[indices1]
    samples2 = sample2[indices2]

    stat_scores1 = stat(samples1, axis=1)
    stat_scores2 = stat(samples2, axis=1)
    stat_scores_diff = stat_scores1 - stat_scores2
    boundaries = np.percentile(stat_scores_diff, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return boundaries


def proportion_confint(sample, alpha=0.05):
    '''Wilson\'s сonfidence interval for a proportion.'''
    p = np.mean(sample)
    n = len(sample)

    z = stats.norm.ppf(1 - alpha / 2)
    left_boundary = 1 / (1 + z ** 2 / n) * (p + z ** 2 / (2 * n) \
                                            - z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)))
    right_boundary = 1 / (1 + z ** 2 / n) * (p + z ** 2 / (2 * n) \
                                             + z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)))

    return left_boundary, right_boundary


def proportions_diff_confint_ind(sample1, sample2, alpha=0.05):
    '''Confidence interval for the difference of two independent proportions.'''
    z = stats.norm.ppf(1 - alpha / 2)
    p1 = np.mean(sample1)
    p2 = np.mean(sample2)
    n1 = len(sample1)
    n2 = len(sample2)

    left_boundary = (p1 - p2) - z * np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    right_boundary = (p1 - p2) + z * np.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)

    return left_boundary, right_boundary


def proportions_diff_confint_rel(sample1, sample2, alpha=0.05):
    '''Confidence interval for the difference of two related proportions.'''
    z = stats.norm.ppf(1 - alpha / 2)
    sample = list(zip(sample1, sample2))
    n = len(sample)

    f = sum([obs[0] == 1 and obs[1] == 0 for obs in sample])
    g = sum([obs[0] == 0 and obs[1] == 1 for obs in sample])

    left_boundary = (f - g) / n - z * np.sqrt((f + g) / n ** 2 - ((f - g) ** 2) / n ** 3)
    right_boundary = (f - g) / n + z * np.sqrt((f + g) / n ** 2 - ((f - g) ** 2) / n ** 3)

    return left_boundary, right_boundary


# -------------------------- Hypotheses testing --------------------------

from scipy.stats import chisquare, shapiro, ttest_1samp, ttest_ind, ttest_rel
from statsmodels.stats.descriptivestats import sign_test
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.sandbox.stats.multicomp import multipletests


def permutation_test(sample, mean, max_permutations=None, alternative='two-sided'):
    '''Permutation test for a sample.'''
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError('Alternative not recognized, should be \'two-sided\', \'less\' or \'greater\'.')

    centered_sample = list(map(lambda x: x - mean, sample))
    t_stat = sum(centered_sample)

    if max_permutations:
        signs_array = set([tuple(x) for x in 2 * np.random.randint(2, size=(max_permutations, len(sample))) - 1])
    else:
        signs_array =  itertools.product([-1, 1], repeat=len(sample))

    zero_distr = [sum(centered_sample * np.array(signs)) for signs in signs_array]

    if alternative == 'two-sided':
        p_value = sum([abs(x) >= abs(t_stat) for x in zero_distr]) / len(zero_distr)

    if alternative == 'less':
        p_value = sum([x <= t_stat for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        p_value = sum([x >= t_stat for x in zero_distr]) / len(zero_distr)

    return t_stat, p_value


def permutation_test_ind(sample1, sample2, max_permutations=None, alternative='two-sided'):
    '''Permutation test for two independent samples.'''
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError('Alternative not recognized, should be \'two-sided\', \'less\' or \'greater\'.')

    t_stat = np.mean(sample1) - np.mean(sample2)

    joined_sample = np.hstack((sample1, sample2))
    n1 = len(sample1)
    n = len(joined_sample)

    if max_permutations:
        index = list(range(n))
        indices = set([tuple(index)])
        for _ in range(max_permutations - 1):
            np.random.shuffle(index)
            indices.add(tuple(index))

        indices = [(index[:n1], index[n1:]) for index in indices]
    else:
        indices = [(list(index), list(filter(lambda i: i not in index, range(n)))) \
                    for index in itertools.combinations(range(n), n1)]

    zero_distr = [joined_sample[list(i[0])].mean() - joined_sample[list(i[1])].mean() \
                  for i in indices]

    if alternative == 'two-sided':
        p_value = sum([abs(x) >= abs(t_stat) for x in zero_distr]) / len(zero_distr)

    if alternative == 'less':
        p_value = sum([x <= t_stat for x in zero_distr]) / len(zero_distr)

    if alternative == 'greater':
        p_value = sum([x >= t_stat for x in zero_distr]) / len(zero_distr)

    return t_stat, p_value


def proportion_ztest(sample, p0=0.5, alternative='two-sided'):
    '''Z-test for a proportion.'''
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError('Alternative not recognized, should be \'two-sided\', \'less\' or \'greater\'.')

    p = np.mean(sample)
    n = len(sample)

    z_stat = (p - p0) / np.sqrt((p0 * (1 - p0)) / n)

    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    if alternative == 'less':
        p_value = stats.norm.cdf(z_stat)

    if alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_stat)

    return z_stat, p_value


def proportions_ztest_ind(sample1, sample2, alternative='two-sided'):
    '''Z-test for two independent proportions.'''
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError('Alternative not recognized, should be \'two-sided\', \'less\' or \'greater\'.')

    p1 = np.mean(sample1)
    p2 = np.mean(sample2)
    n1 = len(sample1)
    n2 = len(sample2)

    P = (p1 * n1 + p2 * n2) / (n1 + n2)
    z_stat = (p1 - p2) / np.sqrt(P * (1 - P) * (1 / n1 + 1 / n2))

    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    if alternative == 'less':
        p_value = stats.norm.cdf(z_stat)

    if alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_stat)

    return z_stat, p_value


def proportions_ztest_rel(sample1, sample2, alternative='two-sided'):
    '''Z-test for two related proportions.'''
    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError('Alternative not recognized, should be \'two-sided\', \'less\' or \'greater\'.')

    sample = list(zip(sample1, sample2))
    n = len(sample)

    f = sum([obs[0] == 1 and obs[1] == 0 for obs in sample])
    g = sum([obs[0] == 0 and obs[1] == 1 for obs in sample])
    z_stat = (f - g) / np.sqrt(f + g - ((f - g) ** 2) / n)

    if alternative == 'two-sided':
        p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))

    if alternative == 'less':
        p_value = stats.norm.cdf(z_stat)

    if alternative == 'greater':
        p_value = 1 - stats.norm.cdf(z_stat)

    return z_stat, p_value


# -------------------------- Correlations --------------------------

from scipy.stats import pearsonr, spearmanr, chi2_contingency, pointbiserialr

def matthews_correlation(contingency_table):
    '''Matthews correlation.'''
    a, b = contingency_table[0]
    c, d = contingency_table[1]

    n = np.sum(contingency_table)
    acabn = (a + c) * (a + b) / n
    accdn = (a + c) * (c + d) / n
    bdabn = (b + d) * (a + b) / n
    bdcdn = (b + d) * (c + d) / n
    if n < 40 or np.any(np.array([acabn, accdn, bdabn, bdcdn]) < 5):
        raise ValueError('Contingency table isn\'t suitable for Matthews correlation calculation.')

    p_value = stats.chi2_contingency(contingency_table)[1]
    corr = (a * d - b * c) / np.sqrt((a + b) * (a + c) * (b + d) * (c + d))
    return corr, p_value


def cramers_v(contingency_table):
    '''Cramer\'s V coefficient.'''
    n = np.sum(contingency_table)
    ct_nrows, ct_ncols = contingency_table.shape
    if n < 40 or np.sum(contingency_table < 5) / (ct_nrows * ct_ncols) > 0.2:
        raise ValueError('Contingency table isn\'t suitable for Cramers\'s V coefficient calculation.')

    chi2, p_value = stats.chi2_contingency(contingency_table)[:2]
    corr = np.sqrt(chi2 / (n * (min(ct_nrows, ct_ncols) - 1)))
    return corr, p_value

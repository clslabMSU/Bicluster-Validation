import numpy as np
import os; os.environ["R_HOME"] = r"C:\Program Files\R\R-3.4.0"; os.environ["R_USER"] = ""
from pickle import dump, load
import rpy2.robjects as robjects

# ====================================== BICLUSTER QUALITY MEASURES ====================================== #

def MSR(bicluster: np.ndarray) -> float:
    """
    Mean Squared Residue Score
    Cheng, Y., & Church, G. M. (2000, August). Biclustering of expression data. In Ismb (Vol. 8, No. 2000, pp. 93-103).

    :param bicluster: - np.ndarray of expression levels of bicluster
    :return: - mean squared residue score, lower is better
    """
    column_means = np.mean(bicluster, axis=0)
    row_means = np.mean(bicluster, axis=1)
    bicluster_mean = np.mean(bicluster.flatten())
    msr = 0
    for i in range(bicluster.shape[0]):
        for j in range(bicluster.shape[1]):
            msr += (bicluster[i, j] - row_means[i] - column_means[j] + bicluster_mean)**2
    return msr / (bicluster.shape[0] * bicluster.shape[1])

def SMSR(bicluster: np.ndarray) -> float:
    """
    Scaled Mean Squared Residue Score
    Mukhopadhyay, A., Maulik, U., & Bandyopadhyay, S. (2009). A novel coherence measure for discovering scaling biclusters from gene expression data. Journal of Bioinformatics and Computational Biology, 7(05), 853-868.

    :param bicluster: - np.ndarray of expression levels of bicluster
    :return: - scaled mean squared residue score, lower is better
    """
    column_means = np.mean(bicluster, axis=0)
    row_means = np.mean(bicluster, axis=1)
    bicluster_mean = np.mean(bicluster.flatten())
    smsr = 0
    for i in range(bicluster.shape[0]):
        for j in range(bicluster.shape[1]):
            smsr += (row_means[i] * column_means[j] - bicluster[i, j] * bicluster_mean)**2 / (row_means[i]**2 * column_means[j]**2)
    return smsr / (bicluster.shape[0] * bicluster.shape[1])

def VE(bicluster: np.ndarray) -> float:
    """
    Virtual Error of a bicluster
    Divina, F., Pontes, B., Giráldez, R., & Aguilar-Ruiz, J. S. (2012). An effective measure for assessing the quality of biclusters. Computers in biology and medicine, 42(2), 245-256.

    :param bicluster: - np.ndarray of expression levels of bicluster
    :return: virtual error score, lower is better
    """
    rho = np.mean(bicluster, axis=0)
    rho_std = np.std(rho)
    if rho_std != 0:
        rho_hat = (rho - np.mean(rho)) / np.std(rho)
    else:
        rho_hat = (rho - np.mean(rho))
    bic_hat = _standardize_bicluster_(bicluster)
    ve = 0
    for i in range(bicluster.shape[0]):
        for j in range(bicluster.shape[1]):
            ve += abs(bic_hat[i, j] - rho_hat[j])
    ve /= (bicluster.shape[0] * bicluster.shape[1])
    return ve

def VEt(bicluster: np.ndarray) -> float:
    return VE(np.transpose(bicluster))

def ASR(bicluster: np.ndarray) -> float or None:
    if bicluster.shape[0] <= 1 or bicluster.shape[1] <= 1:
        return None
    spearman_genes = 0
    spearman_samples = 0
    for i in range(bicluster.shape[0] - 1):
        for j in range(i+1, bicluster.shape[0]):
            spearman_genes += spearman(bicluster[i, :], bicluster[j, :])
    for k in range(bicluster.shape[1] - 1):
        for l in range(k+1, bicluster.shape[1]):
            spearman_samples += spearman(bicluster[:, k], bicluster[:, l])
    spearman_genes /= bicluster.shape[0]*(bicluster.shape[0]-1)
    spearman_samples /= bicluster.shape[1]*(bicluster.shape[1]-1)
    asr = 2*max(spearman_genes, spearman_samples)
    return asr

def spearman(x: np.ndarray, y: np.ndarray) -> float:
    assert(x.shape == y.shape)
    rx = rankdata(x)
    ry = rankdata(y)
    m = len(x)
    coef = 6.0/(m**3-m)
    ans = 0
    for k in range(m):
        ans += (rx[k] - ry[k])**2
    return 1 - coef*ans

def _standardize_bicluster_(bicluster: np.ndarray) -> np.ndarray:
    """
    Standardize a bicluster by subtracting the mean and dividing by standard deviation.
    Pontes, B., Girldez, R., & Aguilar-Ruiz, J. S. (2015). Quality measures for gene expression biclusters. PloS one, 10(3), e0115497.

    Note that UniBic synthetic data was generated with mean 0 and standard deviation 1, so it is already standardized.

    :param bicluster: np.ndarray of expression levels of bicluster
    :return: standardized bicluster
    """
    bic = np.copy(bicluster)
    for i in range(bic.shape[0]):
        gene = bic[i, :]
        std = np.std(gene)
        if std != 0:
            bic[i, :] = (gene - np.mean(gene)) / std
        else:
            bic[i, :] = (gene - np.mean(gene))
    return bic

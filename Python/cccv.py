from data import *
from sklearn.calibration        import CalibratedClassifierCV
from sklearn.ensemble           import ExtraTreesClassifier
from sklearn.cross_validation   import StratifiedKFold
from scipy.stats                import randint
import logging

logging.basicConfig(format="[%(asctime)s] %(levelname)s\t%(message)s",
                filename="history.log",
                filemode='a', level=logging.DEBUG,
                datefmt='%m/%d/%y %H:%M:%S')
formatter = logging.Formatter("[%(asctime)s] %(levelname)s\t%(message)s",
                datefmt='%m/%d/%y %H:%M:%S')
console = logging.StreamHandler()
console.setFormatter(formatter)
console.setLevel(logging.INFO)
logging.getLogger().addHandler(console)
logger = logging.getLogger(__name__)
logger = logging.getLogger(__name__)

X, y, Xtest = LoadData()
X = X[:1000]
y = y[:1000]
np.random.seed(0)
for i in xrange(100):
    kcv = StratifiedKFold(y, n_folds = 5)
    max_features = randint(10,93).rvs()
    n_estimators = randint(100, 1000).rvs()
    res = np.zeros((len(y), 9)); j = 1
    for train_idx, valid_idx in kcv:
        logger.info('%4d|%4d|%4d', max_features, n_estimators, j); j += 1
        clf = CalibratedClassifierCV(
                base_estimator  = ExtraTreesClassifier(
                    n_estimators    = n_estimators, 
                    max_features    = max_features, 
                    verbose         = 0, 
                    n_jobs          = -1), 
                method          = 'isotonic',
                cv              = 10)
        clf.fit(X[train_idx], y[train_idx])
        res[valid_idx] = clf.predict_proba(X[valid_idx])
    logger.info('%8.4f', log_loss(y, res))

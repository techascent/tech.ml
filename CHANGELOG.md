# ConstantChangeLog

## 2.0-beta-56

### **Breaking Change:** Smile 2.4.0 upgrade.
There are fewer smile regressors and classifiers supported.  XGBoost support is the
same.  This requires `[techascent/tech.ml.dataset "2.0-beta-56"]` or later.  If you
are using dataset, xgboost, or the set of supported smile regressors and classifiers
changes to your code should be zero or minimal.


## 2.0-beta-48
**Breaking Change:** This library expects tech.ml.dataset to be provided.  So your project
needs both this library and `[tech.ml.dataset "2.0-beta-49"]`.  This is to reduce the
number of spurious releases.

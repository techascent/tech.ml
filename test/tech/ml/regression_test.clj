(ns tech.ml.regression-test
  (:require [tech.ml.regression :as ml-reg]
            [tech.ml.dataset :as ds]
            [tech.ml.visualization.vega :as vega-viz]
            [tech.verify.ml.regression :as verify-reg]
            [clojure.test :refer :all]
            [oz.core :as oz]))


(deftest shotgun-approach-sanity
  (testing "Simple shotgun approach produces results"
    (let [{:keys [train-ds test-ds options]} (verify-reg/datasets)
          dataset (ds/ds-concat train-ds test-ds)
          trained-models (ml-reg/train-regressors dataset options)
          model-viz (->> (concat [(vega-viz/accuracy-graph
                                   (map
                                    #(dissoc % :labels :predictions :residuals :model)
                                    trained-models))]
                                 (->> trained-models
                                      (map (fn [trained-model]
                                             [(vega-viz/graph-regression-verification-results
                                               trained-model :target-key :predictions)
                                              (vega-viz/graph-regression-verification-results
                                               trained-model :target-key :residuals)]))))
                         (into [:div]))]
      ;;xgboost, lasso always work.  ridge sometimes and libsvm only if
      ;;libsvm is actually installed.
      (is (>= (count trained-models) 2)))))



(deftest regressors-regression
  (testing "Simple shotgun approach produces results"
    (let [{:keys [train-ds test-ds options]} (verify-reg/datasets)
          dataset (ds/ds-concat train-ds test-ds)
          trained-models (ml-reg/train-regressors
                          dataset options
                          :gridsearch-regression-systems [:smile.regression/lasso
                                                          :xgboost/regression])
          model-viz (->> (concat [(vega-viz/accuracy-graph
                                   (map
                                    #(dissoc % :labels :predictions :residuals :model)
                                    trained-models))]
                                 (->> trained-models
                                      (map (fn [trained-model]
                                             [(vega-viz/graph-regression-verification-results
                                               trained-model :target-key :predictions)
                                              (vega-viz/graph-regression-verification-results
                                               trained-model :target-key :residuals)]))))
                         (into [:div]))]
      ;;xgboost, lasso always work.  ridge sometimes and libsvm only if
      ;;libsvm is actually installed.
      (is (>= (count trained-models) 2)))))

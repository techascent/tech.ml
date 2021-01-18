(ns tech.v3.libs.xgboost-test
  (:require [clojure.test :refer [deftest is]]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.functional :as dfn]
            [tech.v3.libs.smile.discrete-nb :as nb]
            [tech.v3.libs.smile.nlp :as nlp]
            [tech.v3.libs.xgboost]
            [tech.v3.ml :as ml]
            [tech.v3.ml.metrics :as metrics]
            [tech.v3.ml.loss :as loss]
            [tech.v3.ml.verify :as verify]
            [tech.v3.ml.classification :as ml-class]
            [tech.v3.libs.xgboost]))

(deftest basic
  (verify/basic-regression {:model-type :xgboost/regression}))


(deftest basic-early-stopping
  (verify/basic-regression {:model-type :xgboost/regression
                            :early-stopping-round 5
                            :round 50}))


(deftest watches
  (let [{train-dataset :train-ds
         test-dataset :test-ds} (ds-mod/train-test-split @verify/regression-iris*)
        options {:model-type :xgboost/regression
                 :watches {:test-ds test-dataset}
                 :round 25
                 :eval-metric "mae"}
        model (ml/train train-dataset options)
        watch-data (get-in model [:model-data :metrics])
        predictions (ml/predict test-dataset model)
        mse (loss/mse (predictions verify/target-colname)
                      (test-dataset verify/target-colname))]
    (is (= 25 (ds/row-count watch-data)))
    (is (not= 0 (dfn/reduce-+ (watch-data :test-ds))))
    (is (< mse (double 0.2)))))


(deftest k-fold-regression
  (verify/k-fold-regression {:model-type :xgboost/regression}))


(deftest regression-gridsearch
  (verify/auto-gridsearch-regression {:model-type :xgboost/regression}))


(deftest classification
  (verify/basic-classification {:model-type :xgboost/classification}))


(deftest classification-gridsearch
  (verify/auto-gridsearch-classification {:model-type :xgboost/classification}))


(deftest sparse-train-does-not-crash []
    (let [reviews
          (->
           (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword })
           (ds/select-columns [:Text :Score])
           (nlp/count-vectorize :Text :bow nlp/default-text->bow)
           (nb/bow->SparseArray :bow :bow-sparse  #(nlp/->vocabulary-top-n % 100))
           (ds/drop-columns [:Text :bow])
           (ds/update-column :Score
                             (fn [col]
                               (let [val-map {0 :c0
                                              1 :c1
                                              2 :c2
                                              3 :c3
                                              4 :c4
                                              5 :c5}]
                                 (dtype/emap val-map :keyword col))))
           (ds/categorical->number cf/categorical)
           (ds-mod/set-inference-target :Score))
          folds
          (ml/train-k-fold reviews {:model-type :xgboost/classification
                                    :sparse-column :bow-sparse
                                    :n-sparse-columns 100
                                    })
          explanation (ml/explain folds)
          ]))

(comment
  (def reviews

    (->
     (ds/->dataset "test/data/reviews.csv.gz" {:key-fn keyword })
     (ds/select-columns [:Text :Score])
     (nlp/count-vectorize :Text :bow nlp/default-text->bow)
     (nb/bow->SparseArray :bow :bow-sparse  #(nlp/->vocabulary-top-n % 100))
     (ds/drop-columns [:Text :bow])
     (ds/update-column :Score
                       (fn [col]
                         (let [val-map {0 :c0
                                        1 :c1
                                        2 :c2
                                        3 :c3
                                        4 :c4
                                        5 :c5}]
                           (dtype/emap val-map :keyword col))))
     (ds/categorical->number cf/categorical)
     (ds-mod/set-inference-target :Score))
    )
  (def trained-model
    (ml/train reviews {:model-type :xgboost/classification
                       :sparse-column :bow-sparse
                       :n-sparse-columns 100
                       :silent 0
                       :round 1
                       :eval-metric "merror"
                       :watches {:test-ds (ds/sample  reviews 10)}
                       }))


  (def prediction
    (:Score
     (ml/predict reviews trained-model )))

  (metrics/accuracy (:Score reviews) prediction )

  (def folds
    (ml/train-k-fold reviews {:model-type :xgboost/classification
                              :sparse-column :bow-sparse

                              }))
  (ml/explain folds))

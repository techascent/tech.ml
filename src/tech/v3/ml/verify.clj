(ns tech.v3.ml.verify
  (:require [tech.v3.datatype.functional :as dfn]
            [tech.v3.datatype :as dtype]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.categorical :as ds-cat]
            [tech.v3.ml :as ml]
            [tech.v3.ml.loss :as loss]
            [clojure.test :refer [is]]))


(def target-colname "petal_width")


(def regression-iris* (delay
                        (-> (ds/->dataset "test/data/iris.csv")
                            (ds/remove-column "species")
                            (ds-mod/set-inference-target "petal_width"))))


(def classification-titanic* (delay
                               (-> (ds/->dataset "test/data/titanic.csv")
                                   (ds/remove-column "Name")
                                   ;;We have to have a lookup map for the column in order to
                                   ;;do classification on the column.
                                   (ds/update-column "Survived"
                                                     (fn [col]
                                                       (let [val-map {0 :drowned
                                                                      1 :survived}]
                                                         (dtype/emap val-map :keyword col))))
                                   (ds/categorical->number cf/categorical)
                                   (ds-mod/set-inference-target "Survived"))))


(defn basic-regression
  ([options-map max-avg-loss]
   (let [train-fn #(ml/train-split @regression-iris* options-map)
         avg-mae (->> (repeatedly 5 train-fn)
                      (map :loss)
                      (dfn/mean))]
     (is (< avg-mae max-avg-loss))))
  ([options-map]
   (basic-regression options-map 0.5)))


(defn k-fold-regression
  ([options-map max-avg-loss]
   (let [model (ml/train-k-fold @regression-iris* options-map)]
     (is (< (double (:avg-loss model))
            (double max-avg-loss)))))
  ([options-map]
   (k-fold-regression options-map 0.5)))


(defn auto-gridsearch-regression
  ([options-map max-avg-loss]
   (let [model (first (ml/train-auto-gridsearch @regression-iris* options-map))]
     (is (< (double (:avg-loss model))
            (double max-avg-loss)))))
  ([options-map]
   (auto-gridsearch-regression options-map 0.5)))


(defn basic-classification
  ([options-map max-avg-loss]
   (let [train-fn #(ml/train-split @classification-titanic* options-map)
         avg-mae (->> (repeatedly 5 train-fn)
                      (map :loss)
                      (dfn/mean))]
     (is (< avg-mae max-avg-loss))))
  ([options-map]
   (basic-classification options-map 0.5)))


(defn auto-gridsearch-classification
  ([options-map max-avg-loss]
   (let [{:keys [avg-loss]}
         (-> (ml/train-auto-gridsearch @classification-titanic* options-map)
             (first))]
     (is (< avg-loss max-avg-loss))))
  ([options-map]
   (basic-classification options-map 0.5)))

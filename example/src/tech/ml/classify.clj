(ns tech.ml.classify
  (:require [tech.ml.svm-datasets :as datasets]
            [tech.ml :as ml]
            [tech.ml.dataset :as ml-dataset]
            [tech.ml.loss :as loss]
            [tech.libs.xgboost]
            [tech.libs.smile.classification]
            [tech.libs.svm]
            [clojure.core.matrix :as m]
            [taoensso.nippy :as nippy]
            [tech.io :as io]))


(defn gridsearch-the-things
  []
  (let [base-systems [{:model-type :xgboost/classification}
                      {:model-type :smile.classification/svm}
                      {:model-type :smile.classification/knn}
                      {:model-type :smile.classification/ada-boost}
                      {:model-type :smile.classification/logistic-regression}]]
    (->> (datasets/all-datasets)
         (map (fn [{:keys [train-ds test-ds name options] :as base-dataset}]
                ;;We use k-fold so for now just merge everything.
                {:name name
                 :top-models
                 (let [train-ds (if test-ds
                                  (ml-dataset/ds-concat train-ds test-ds)
                                  train-ds)]
                   (->> base-systems
                        (mapcat (fn [opts]
                                  (let [gs-options (ml/auto-gridsearch-options opts)]
                                    (println (format "Dataset: %s, Model: %s" name (:model-type opts)))
                                    (->> (ml/gridsearch (merge options gs-options
                                                               {:k-fold (if (> (first (m/shape train-ds)) 200)
                                                                          5
                                                                          3)
                                                                :gridsearch-depth 75})
                                                        loss/classification-loss
                                                        train-ds)
                                         (map (partial merge (dissoc base-dataset :test-ds :train-ds :options)))))))
                        (sort-by :average-loss)
                        (take 15)))})))))



(defn large-gridsearch-summary
  [gridsearch-results]
  (->> gridsearch-results
       (map (fn [{:keys [name top-models]}]
              {:name name
               :top-models (->> top-models
                                (map (fn [model-result]
                                       (update model-result
                                               :options select-keys [:model-type]))))}))))


(defn doit
  []
  (let [results (gridsearch-the-things)
        summary (large-gridsearch-summary results)]
    (io/put-nippy! "file://results.nippy" results)
    summary))

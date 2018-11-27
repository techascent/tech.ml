(ns tech.ml.classify
  (:require [tech.ml.svm-datasets :as datasets]
            [tech.ml-base :as ml]
            [tech.ml.dataset :as ml-dataset]
            [tech.ml.loss :as loss]
            [tech.xgboost]
            [tech.svm]
            [tech.smile.classification]
            [taoensso.nippy :as nippy]
            [tech.io :as io]))


(defn gridsearch-the-things
  []
  (let [base-systems [{:system-name :xgboost
                       :model-type :classification}
                      {:system-name :smile/classification
                       :model-type :svm}
                      {:system-name :smile/classification
                       :model-type :knn}
                      {:system-name :smile/classification
                       :model-type :ada-boost}
                      {:system-name :libsvm
                       :model-type :classification}]]
    (->> (datasets/all-datasets)
         (map (fn [{:keys [train-ds test-ds name] :as base-dataset}]
                ;;We use k-fold so for now just merge everything.
                {:name name
                 :top-models
                 (let [train-ds (->> (if test-ds
                                       (concat test-ds train-ds)
                                       train-ds)
                                     shuffle
                                     (take 2000)
                                     vec)]
                   (->> base-systems
                        (mapcat (fn [{:keys [system-name] :as opts}]
                                  (let [gs-options (ml/auto-gridsearch-options system-name opts)]
                                    (println (format "Dataset: %s, System: %s, Model %s" name system-name
                                                     (:model-type opts)))
                                    (->> (ml/gridsearch [[system-name gs-options]]
                                                        :features :label loss/classification-loss train-ds
                                                        :k-fold 3
                                                        :range-map {::ml-dataset/features [-1 1]}
                                                        :gridsearch-depth 75)
                                         (map #(merge (dissoc base-dataset :test-ds :train-ds)
                                                      %))))))
                        (sort-by :error)
                        (take 10)))})))))



(defn large-gridsearch-summary
  [gridsearch-results]
  (->> gridsearch-results
       (map (fn [{:keys [name top-models]}]
              {:name name
               :top-models (->> top-models
                                (map (fn [model-result]
                                       (update model-result
                                               :options select-keys [:system-name :model-type]))))}))))

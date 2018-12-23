(ns tech.ml.classify
  (:require [tech.ml.svm-datasets :as datasets]
            [tech.ml-base :as ml]
            [tech.ml.dataset :as ml-dataset]
            [tech.ml.loss :as loss]
            [tech.xgboost]
            [tech.smile.classification]
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
         (map (fn [{:keys [train-ds test-ds name] :as base-dataset}]
                ;;We use k-fold so for now just merge everything.
                {:name name
                 :top-models
                 (let [train-ds (->> (if test-ds
                                       (concat test-ds train-ds)
                                       train-ds)
                                     shuffle
                                     vec)]
                   (->> base-systems
                        (mapcat (fn [opts]
                                  (let [gs-options (ml/auto-gridsearch-options opts)]
                                    (println (format "Dataset: %s, Model: %s" name (:model-type opts)))
                                    (->> (ml/gridsearch [gs-options]
                                                        :features :label loss/classification-loss train-ds
                                                        :k-fold (if (> (count train-ds) 200)
                                                                  5
                                                                  3)
                                                        ;;Ensure all features are in range -1 to 1.  This is done
                                                        ;;before k-folding so we do get the entire range of the
                                                        ;;dataset.
                                                        :range-map {::ml-dataset/features [-1 1]}
                                                        :gridsearch-depth 75)
                                         (map #(merge (dissoc base-dataset :test-ds :train-ds)
                                                      %))))))
                        (sort-by :average-loss)
                        (take 10)))})))))



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

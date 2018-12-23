(ns tech.ml.train-test
  (:require [clojure.test :refer :all]
            [tech.ml.train :as train]
            [tech.ml.loss :as loss]))


(deftest ds->m-seq
  (is (= (list  {:model {:a 1, :b 2, :c 0}}
                {:model {:a 1, :b 2, :c 1}}
                {:model {:a 1, :b 2, :c 2}}
                {:model {:a 1, :b 2, :c 3}})
         (->> (train/dataset-seq->dataset-model-seq
               #(merge {:a 1 :b 2}
                       (first %))
               (->> (range 4)
                    (mapv (comp #(hash-map :train-ds (repeat 4 %))
                                (partial hash-map :c)))))
              (map #(dissoc % :train-time))))))


(deftest ave-pred-error
  (let [train-fn #(merge {:a 1 :b 2}
                         (first %))
        predict-fn (fn [model dataset]
                     (repeat (count dataset)
                             (:c model)))
        loss-fn loss/mse
        dataset-seq (->> (range 4)
                         (mapv (comp #(hash-map :train-ds (repeat 4 %)
                                                :test-ds (repeat 4 %))
                                     (partial hash-map :c))))]
    (train/average-prediction-error train-fn predict-fn :c loss-fn dataset-seq)))

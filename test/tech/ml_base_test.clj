(ns tech.ml-base-test
  (:require [clojure.test :refer :all]
            [tech.ml-base :as ml-base]
            [tech.ml.protocols :as ml-proto]
            [tech.ml.registry :as registry]
            [tech.ml.dataset :as dataset]
            [tech.ml.loss :as loss]
            [tech.ml.gridsearch :as ml-gs]))


(defrecord TestSystem []
  ml-proto/PMLSystem
  (system-name [system] :test-system)
  (coalesce-options [system options] {})
  (gridsearch-options [system options] {})
  (train [system options coalesced-dataset]
    (Thread/sleep (* 10 (:a-opt options)))
    (assoc options :c (-> (first coalesced-dataset)
                          ::dataset/label
                          first)))
  (predict [system options model coalesced-dataset]
    (Thread/sleep (* 20 (:b-opt options)))
    (repeat (count coalesced-dataset) (:c model))))


(def system (constantly (->TestSystem)))


(registry/register-system (system))



(deftest gridsearch-test
  (let [dataset (->> (range 100)
                     (map (fn [idx]
                            {:a 1 :b idx :c 4})))
        feature-keys [:a :b]
        label-keys :c
        system-name->opt-seq [[:test-system {:a-opt 1 :b-opt 2}]
                              [:test-system {:a-opt 2 :b-opt 3}]] ]

    (is (= (list {:average-loss 0.0,
                  :k-fold 5,
                  :options {:a-opt 1,
                            :b-opt 2,
                            :tech.ml.dataset/dataset-info {:tech.ml.dataset/feature-ecount 2,
                                                           :tech.ml.dataset/key-ecount-map {:a 1,
                                                                                            :b 1,
                                                                                            :c 1}},
                            :tech.ml.dataset/feature-keys [:a :b],
                            :tech.ml.dataset/label-keys [:c]},
                  :system :test-system,}
                 {:average-loss 0.0,
                  :k-fold 5,
                  :options {:a-opt 2,
                            :b-opt 3,
                            :tech.ml.dataset/dataset-info {:tech.ml.dataset/feature-ecount 2,
                                                           :tech.ml.dataset/key-ecount-map {:a 1,
                                                                                            :b 1,
                                                                                            :c 1}},
                            :tech.ml.dataset/feature-keys [:a :b],
                            :tech.ml.dataset/label-keys [:c]},
                  :system :test-system})
           (->> (ml-base/gridsearch system-name->opt-seq feature-keys label-keys loss/mse dataset)
                (map #(dissoc % :train-time :predict-time)))))))

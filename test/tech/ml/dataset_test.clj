(ns tech.ml.dataset-test
  (:require [clojure.test :refer :all]
            [tech.ml.dataset :as dataset]))


(defn- vectorize-result
  [coalesced-ds]
  (->> coalesced-ds
       (mapv (fn [ds-entry]
               (->> ds-entry
                    (map (fn [[k v]]
                           [k (if (number? v)
                                (long v)
                                (mapv long v))]))
                    (into {}))))))


(defn- make-test-ds
  [& [num-items]]
  (->> (range)
       (partition 4)
       (map (fn [item-seq]
              {:a (take 2 item-seq)
               :b (nth item-seq 2)
               :c (last item-seq)}))
       (take (or num-items 10))))


(deftest dataset-base
  (let [test-ds (make-test-ds)]
    (testing "basic coalescing"
      (let [correct [{:values [0 1 2], :label [3]}
                     {:values [4 5 6], :label [7]}
                     {:values [8 9 10], :label [11]}
                     {:values [12 13 14], :label [15]}
                     {:values [16 17 18], :label [19]}
                     {:values [20 21 22], :label [23]}
                     {:values [24 25 26], :label [27]}
                     {:values [28 29 30], :label [31]}
                     {:values [32 33 34], :label [35]}
                     {:values [36 37 38], :label [39]}]
            correct-scalar-label
            [{:values [0 1 2], :label 3}
             {:values [4 5 6], :label 7}
             {:values [8 9 10], :label 11}
             {:values [12 13 14], :label 15}
             {:values [16 17 18], :label 19}
             {:values [20 21 22], :label 23}
             {:values [24 25 26], :label 27}
             {:values [28 29 30], :label 31}
             {:values [32 33 34], :label 35}
             {:values [36 37 38], :label 39}]
            correct-no-label [{:values [0 1 2]}
                              {:values [4 5 6]}
                              {:values [8 9 10]}
                              {:values [12 13 14]}
                              {:values [16 17 18]}
                              {:values [20 21 22]}
                              {:values [24 25 26]}
                              {:values [28 29 30]}
                              {:values [32 33 34]}
                              {:values [36 37 38]}]]
        (is (= correct
               (->> (dataset/coalesce-dataset
                     [:a :b] :c {} test-ds)
                    vectorize-result)))
        (is (= correct
               (->> (dataset/coalesce-dataset
                     [:a :b] :c {:batch-size 1} test-ds)
                    vectorize-result)))
        (is (= correct-no-label
               (->> (dataset/coalesce-dataset
                     [:a :b] nil {:keep-extra? false} test-ds)
                    vectorize-result)))))
    (testing "batch coalescing"
      (let [correct [{:values [0 1 2 4 5 6], :label [3 7]}
                     {:values [8 9 10 12 13 14], :label [11 15]}
                     {:values [16 17 18 20 21 22], :label [19 23]}
                     {:values [24 25 26 28 29 30], :label [27 31]}
                     {:values [32 33 34 36 37 38], :label [35 39]}]]
          (is (= correct
                 (->> (dataset/coalesce-dataset
                       [:a :b] :c {:batch-size 2} test-ds)
                      vectorize-result)))))))

(ns tech.ml.dataset-test
  (:require [clojure.test :refer :all]
            [tech.ml.dataset :as dataset]
            [tech.verify.ml.classification :as vf-classify]
            [clojure.core.matrix :as m]))


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


(defn- vectorize-double-result
  [coalesced-ds]
  (->> coalesced-ds
       (mapv (fn [ds-entry]
               (->> ds-entry
                    (map (fn [[k v]]
                           [k (if (number? v)
                                (double v)
                                (mapv double v))]))
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
      (let [correct [{::dataset/features [0 1 2], ::dataset/label [3]}
                     {::dataset/features [4 5 6], ::dataset/label [7]}
                     {::dataset/features [8 9 10], ::dataset/label [11]}
                     {::dataset/features [12 13 14], ::dataset/label [15]}
                     {::dataset/features [16 17 18], ::dataset/label [19]}
                     {::dataset/features [20 21 22], ::dataset/label [23]}
                     {::dataset/features [24 25 26], ::dataset/label [27]}
                     {::dataset/features [28 29 30], ::dataset/label [31]}
                     {::dataset/features [32 33 34], ::dataset/label [35]}
                     {::dataset/features [36 37 38], ::dataset/label [39]}]
            correct-scalar-label
            [{::dataset/features [0 1 2], ::dataset/label 3}
             {::dataset/features [4 5 6], ::dataset/label 7}
             {::dataset/features [8 9 10], ::dataset/label 11}
             {::dataset/features [12 13 14], ::dataset/label 15}
             {::dataset/features [16 17 18], ::dataset/label 19}
             {::dataset/features [20 21 22], ::dataset/label 23}
             {::dataset/features [24 25 26], ::dataset/label 27}
             {::dataset/features [28 29 30], ::dataset/label 31}
             {::dataset/features [32 33 34], ::dataset/label 35}
             {::dataset/features [36 37 38], ::dataset/label 39}]
            correct-no-label [{::dataset/features [0 1 2]}
                              {::dataset/features [4 5 6]}
                              {::dataset/features [8 9 10]}
                              {::dataset/features [12 13 14]}
                              {::dataset/features [16 17 18]}
                              {::dataset/features [20 21 22]}
                              {::dataset/features [24 25 26]}
                              {::dataset/features [28 29 30]}
                              {::dataset/features [32 33 34]}
                              {::dataset/features [36 37 38]}]]
        (is (= correct
               (->> (dataset/apply-dataset-options
                     [:a :b] :c {} test-ds)
                    :coalesced-dataset
                    vectorize-result)))
        (is (= correct
               (->> (dataset/apply-dataset-options
                     [:a :b] :c {:batch-size 1} test-ds)
                    :coalesced-dataset
                    vectorize-result)))
        (is (= correct-no-label
               (->> (dataset/apply-dataset-options
                     [:a :b] nil {:keep-extra? false} test-ds)
                    :coalesced-dataset
                    vectorize-result)))))
    (testing "batch coalescing"
      (let [correct [{::dataset/features [0 1 2 4 5 6], ::dataset/label [3 7]}
                     {::dataset/features [8 9 10 12 13 14], ::dataset/label [11 15]}
                     {::dataset/features [16 17 18 20 21 22], ::dataset/label [19 23]}
                     {::dataset/features [24 25 26 28 29 30], ::dataset/label [27 31]}
                     {::dataset/features [32 33 34 36 37 38], ::dataset/label [35 39]}]]
          (is (= correct
                 (->> (dataset/apply-dataset-options
                       [:a :b] :c {:batch-size 2} test-ds)
                      :coalesced-dataset
                      vectorize-result)))))
    (testing "batch coalescing with min-max"
    (let [test-ds (make-test-ds)
          {:keys [coalesced-dataset options]}
          (dataset/apply-dataset-options [:a :b] :c
                                         {:batch-size 2
                                          :range-map {::dataset/features [-1 1]}}
                                         test-ds)
          result (vectorize-double-result coalesced-dataset)
          result-values (mapv ::dataset/features result)
          result-labels (mapv #(mapv long (::dataset/label %)) result)]
      (is (= [[3 7] [11 15] [19 23] [27 31] [35 39]]
             result-labels))
      (is (m/equals result-values
                    [[-1.0 -1.0 -1.0 -0.777 -0.777 -0.777]
                     [-0.555 -0.555 -0.555 -0.333 -0.333 -0.333]
                     [-0.111 -0.111 -0.111 0.111 0.111 0.111]
                     [0.333 0.333 0.333 0.555 0.555 0.555]
                     [0.777 0.777 0.777 1.0 1.0 1.0]]
                    0.001))))))


(deftest test-categorical-data
  (let [test-ds (vf-classify/fruit-dataset)
        {:keys [coalesced-dataset options]}
        (dataset/apply-dataset-options [:color-score :height :mass :width] :fruit-name
                                       {:deterministic-label-map? true
                                        :multiclass-label-base-index 1}
                                       test-ds)]
    (is (= options {:label-map {:fruit-name {:apple 1 :lemon 4
                                             :mandarin 2 :orange 3}}
                    :deterministic-label-map? true
                    :multiclass-label-base-index 1
                    ::dataset/dataset-info {::dataset/feature-ecount 4
                                            ::dataset/num-classes 4
                                            ::dataset/key-ecount-map
                                            {:color-score 1 :height 1 :mass 1
                                             :width 1 :fruit-name 1}}
                    ::dataset/feature-keys [:color-score :height :mass :width]
                    ::dataset/label-keys [:fruit-name]}))
    (is (= [{::dataset/features [0 7 192 8], ::dataset/label [1]}
	   {::dataset/features [0 6 180 8], ::dataset/label [1]}
	   {::dataset/features [0 7 176 7], ::dataset/label [1]}
	   {::dataset/features [0 4 86 6], ::dataset/label [2]}
	   {::dataset/features [0 4 84 6], ::dataset/label [2]}]
           (->> coalesced-dataset
                vectorize-result
                (take 5)
                vec)))
    (let [{:keys [coalesced-dataset]}
          (dataset/apply-dataset-options
           [:color-score :height :mass :width] :fruit-name
           {:label-map {:fruit-name {:apple 4 :lemon 2
                                     :mandarin 3 :orange 1}}}
           test-ds)]
      (is (= [{::dataset/features [0 7 192 8], ::dataset/label [4]}
              {::dataset/features [0 6 180 8], ::dataset/label [4]}
              {::dataset/features [0 7 176 7], ::dataset/label [4]}
              {::dataset/features [0 4 86 6], ::dataset/label [3]}
              {::dataset/features [0 4 84 6], ::dataset/label [3]}]
             (->> coalesced-dataset
                  vectorize-result
                  (take 5)
                  vec))))
    (let [{:keys [options coalesced-dataset]}
          (dataset/apply-dataset-options [:color-score :height :mass
                                          :width :fruit-subtype]
                                         :fruit-name
                                         {:deterministic-label-map? true
                                          :multiclass-label-base-index 1} test-ds)]
      (is (= {:label-map
              {:fruit-subtype
               {:golden-delicious 4
                :unknown 10
                :granny-smith 1
                :braeburn 3
                :spanish-jumbo 6
                :selected-seconds 7
                :mandarin 2
                :cripps-pink 5
                :turkey-navel 8
                :spanish-belsan 9}
               :fruit-name {:apple 1 :mandarin 2
                            :orange 3 :lemon 4}}
              :deterministic-label-map? true
              :multiclass-label-base-index 1
              ::dataset/dataset-info {::dataset/feature-ecount 5
                                      ::dataset/num-classes 4
                                      ::dataset/key-ecount-map {:color-score 1
                                                                :height 1
                                                                :mass 1
                                                                :width 1
                                                                :fruit-subtype 1
                                                                :fruit-name 1}}
              ::dataset/feature-keys [:color-score :height :mass
                             :width :fruit-subtype]
              ::dataset/label-keys [:fruit-name]}
             options))
      (is (= [{::dataset/features [0 7 192 8 1], ::dataset/label [1]}
              {::dataset/features [0 6 180 8 1], ::dataset/label [1]}
              {::dataset/features [0 7 176 7 1], ::dataset/label [1]}
              {::dataset/features [0 4 86 6 2], ::dataset/label [2]}
              {::dataset/features [0 4 84 6 2], ::dataset/label [2]}]
             (->> coalesced-dataset
                  vectorize-result
                  (take 5)
                  vec)))
      (testing "Categorical data with scaling"
        (let [{:keys [options coalesced-dataset]}
              (dataset/apply-dataset-options [:color-score :height :mass
                                              :width :fruit-subtype]
                                             :fruit-name
                                             {:deterministic-label-map? true
                                              :multiclass-label-base-index 1
                                              :range-map {::dataset/features [-1 1]}}
                                             test-ds)]
          (is (m/equals
               [[-1.0 0.0153 -0.188 0.368 -1.0]
                [-0.789 -0.138 -0.272 0.157 -1.0]
                [-0.736 -0.0153 -0.300 -0.157 -1.0]
                [0.315 -0.784 -0.930 -0.789 -0.777]
                [0.263 -0.815 -0.944 -0.894 -0.777]]
               (->> coalesced-dataset
                    vectorize-double-result
                    (map ::dataset/features)
                    (take 5)
                    vec)
               0.001)))))))

(ns tech.ml.vega
  (:require [clojure.string :as s]
            [clojure.set :as c-set]
            [tech.ml-base :as ml]))


(defn fixup-model-type
  "Full model type strings are too long for normal legends."
  [gs-result]
  (assoc gs-result
         :model-name
         (-> (get-in gs-result [:options :model-type])
             str
             (s/replace "smile.classification" "smile")
             (s/replace "smile.regression" "smile"))))


(defn accuracy-graph
  [gridsearch-results]
  {:repeat {:column [:predict-time :train-time]}
   :spec {:data {:values (map fixup-model-type gridsearch-results)}
          :mark :point
          :encoding {:y {:field :average-accuracy
                         :type :quantitative}
                     :x {:field {:repeat :column}
                         :type :quantitative}
                     :color {:field :model-name
                             :type :nominal}
                     :shape {:field :model-name
                             :type :nominal}}}})


(defn add-gridsearch-keys
  "group items by model type and add keys which were gridsearched to each item"
  [gridsearch-data]
  (->> gridsearch-data
       (group-by #(get-in % [:options :model-type]))
       (map (fn [[model-type entries]]
              (let [gs-keys (-> (ml/auto-gridsearch-options {:model-type model-type})
                                keys
                                set
                                (disj :model-type))]
                [model-type (mapv #(assoc % :gridsearch-keys gs-keys) entries)])))
       (into {})))


(defn gridsearch-hyperopt-graphs
  "Given a set of gridsearch results and a key for the x axis, see how the
  hyperparams affect the results.  Returns a sequence of
  [model-type vega-lite-graph-specification-seq]"
  [x-axis-key gridsearch-data]
  (->> gridsearch-data
       (mapv (fn [[model-type entries]]
               (let [fs-opts (->> (:gridsearch-keys (first entries))
                                  sort)
                     data
                     (->> entries
                          (map (fn [entry]
                                 (merge entry
                                        (select-keys (:options entry) fs-opts)
                                        {:kernel-type (get-in entry
                                                              [:options :kernel
                                                               :kernel-type])}))))
                     fs-opt->datatype-map
                     (->> fs-opts
                          (map (fn [fs-opt]
                                 [fs-opt (if (number?
                                              (get (first data) fs-opt))
                                           :quantitative
                                           :nominal)]))
                          (into {}))
                     nominal-opts (->> fs-opt->datatype-map
                                       (filter #(= :nominal (second %)))
                                       (map first)
                                       set)
                     quantitative-opts (c-set/difference (:gridsearch-keys
                                                          (first entries))
                                                         nominal-opts)]
                 [model-type
                  (concat
                   (->> quantitative-opts
                        (partition-all 2)
                        (map (fn [op-seq]
                               {:repeat {:column (mapv name (sort op-seq))}
                                :spec {:data {:values data}
                                       :mark :point
                                       :encoding {:x {:field x-axis-key
                                                      :type :quantitative}
                                                  :y {:field {:repeat :column}
                                                      :type :quantitative}}}})))
                   (->> nominal-opts
                        (partition-all 2)
                        (map (fn [op-seq]
                               {:repeat {:column (->> op-seq
                                                      (map (fn [ktype]
                                                             (if (= ktype :kernel)
                                                               :kernel-type
                                                               ktype)))
                                                      sort
                                                      (mapv name))}
                                :spec {:data {:values data}
                                       :mark :point
                                       :encoding {:x {:field x-axis-key
                                                      :type :quantitative}
                                                  :y {:field {:repeat :column}
                                                      :type :nominal}}}}))))])))))

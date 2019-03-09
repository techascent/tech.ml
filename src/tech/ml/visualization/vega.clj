(ns tech.ml.visualization.vega
  (:require [clojure.string :as s]
            [clojure.set :as c-set]
            [tech.ml :as ml]))


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
  [gridsearch-results & {:keys [x-scale y-scale]}]
  [:vega-lite (merge {:data {:values (map fixup-model-type gridsearch-results)}
                      :mark :point
                      :encoding {:y (merge {:field :average-loss
                                            :type :quantitative}
                                           (when y-scale
                                             {:scale {:domain y-scale}}))
                                 :x {:field :model-name
                                     :type :nominal}
                                 :color {:field :model-name
                                         :type :nominal}
                                 :shape {:field :model-name
                                         :type :nominal}}}
                     (when y-scale
                       {:transform [{:filter {:field :average-loss :range y-scale}}]}))])


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


(defn graph-regression-verification-results
  "There are two possible target keys, predictions or residuals.
  These are graphed against labels (labels on the y axis).  Custom scales
  may be used and only results within the range of those scales will be displaced
  and the graph will be clipped to that scale."
  [verification-results & {:keys [target-key x-scale y-scale]
                           :or {target-key :predictions}}]
  (when-not (and (contains? verification-results :labels)
                 (contains? verification-results target-key))
    (throw (ex-info "Results do not appear to contain correct information."
                    {:result-keys (keys verification-results)})))
  (let [graph-data (map (fn [label arg]
                          {:labels label
                           target-key arg})
                        (get verification-results :labels)
                        (get verification-results target-key))]
    [:vega-lite (merge {:data {:values graph-data}
                        :mark :point
                        :encoding {:y (merge {:field :labels
                                              :type :quantitative}
                                             (when y-scale
                                               {:scale {:domain y-scale}}))
                                   :x (merge {:field target-key
                                              :type :quantitative}
                                             (when x-scale
                                               {:scale {:domain x-scale}}))}}
                       (when (or x-scale y-scale)
                         {:transform (concat (when y-scale
                                               [{:filter {:field :labels :range y-scale}}])
                                             (when x-scale
                                               [{:filter {:field target-key :range x-scale}}]))}))]))

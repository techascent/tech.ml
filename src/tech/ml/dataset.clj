(ns tech.ml.dataset
  "The most simple dataset description we have figured out is a sequence of maps.

  Using this definition, things like k-fold have natural interpretations.

  While this works well for clojure generation/manipulation,  ML interfaces uniformly
  require this sequence to be coalesced somehow into larger buffers; sometimes on a
  point by point basis and sometimes into batching buffers.  This file is intended
  to provide direct, simple tools to provide either type of coalesced information.

  Care has been taken to keep certain operations lazy so that datasets of unbounded
  length can be manipulated.  Operatings like auto-scaling, however, will read the
  dataset into memory."
  (:require [tech.datatype :as dtype]
            [tech.parallel :as parallel]
            [tech.compute.cpu.tensor-math :as cpu-tm]
            [tech.compute.tensor :as ct]
            [tech.compute.tensor.operations :as ops]
            [clojure.core.matrix.stats :as m-stats]
            [tech.ml.utils :as utils])
  (:import [java.util Iterator NoSuchElementException]))


(set! *warn-on-reflection* true)


(defn get-dataset-item
  [dataset-entry item-key {:keys [label-map]}]
  (if-let [retval (get dataset-entry item-key)]
    (if-let [retval-labels (get label-map item-key)]
      (if-let [labelled-retval (get retval-labels retval)]
        labelled-retval
        (throw (ex-info "Failed to find label for item"
                        {:item-key item-key
                         :ds-entry retval
                         :label-map label-map})))
      retval)
    (throw (ex-info "Failed to get dataset item"
                    {:item-key item-key
                     :dataset-entry-keys (keys dataset-entry)}))))


(defn- ecount-map
  [entry-keys dataset-entry-values]
  (->> (map dtype/ecount dataset-entry-values)
       (zipmap entry-keys)))


(defn- dataset-entry->data
  [entry-keys dataset-entry options]
  (->> entry-keys
       (map #(get-dataset-item dataset-entry % options))))


(defn normalize-keys
  [kwd-or-seq]
  (when kwd-or-seq
    (-> (if-not (sequential? kwd-or-seq)
          [kwd-or-seq]
          kwd-or-seq)
        vec)))


(defn- check-entry-ecounts
  [expected-ecount-map all-keys ds-entry options]
  (let [entry-values (dataset-entry->data all-keys ds-entry options)
        ds-entry-ecounts (ecount-map all-keys entry-values)]
    (when-not (= expected-ecount-map
                 ds-entry-ecounts)
      (throw (ex-info "Dataset ecount mismatch"
                      {:first-item-ecounts expected-ecount-map
                       :entry-ecounts ds-entry-ecounts})))
    entry-values))


(defn- ecount-from-map
  [key-seq ecount-map]
  (->> key-seq
       (map #(get ecount-map %))
       (apply + 0)))


(defn- dataset->batched-dataset
  "Partition a dataset into batches and produce a sequence of maps
  that include the information from the original dataset but the
  features and labels are concatenated on a key-by-key basis.
  dataset keys not in data-keys are all coalesced into
  :extra-data.

  Options
  :keep-extra? Keep the extra data or discard it.
  :ensure-commensurate? Ensure all batches are of same size.

  Returns
  {::features ::label :extra-data}"
  [feature-keys label-keys batch-size {:keys [keep-extra? label-map]
                                       :or {keep-extra? true}
                                       :as options}
   dataset]
  (let [feature-keys (normalize-keys feature-keys)
        label-keys (normalize-keys label-keys)
        n-features (count feature-keys)
        all-keys (concat feature-keys label-keys)
        expected-ecount-map (->> (dataset-entry->data all-keys (first dataset) options)
                                 (ecount-map all-keys))
        batch-size (or batch-size 1)]
    ;;We have to remember the ecounts at a high level because once they are
    ;;interleaved we get potentially ragged dimensions and there is no efficient
    ;;way to handle that.
    {:value-ecount (* batch-size
                      (ecount-from-map feature-keys expected-ecount-map))
     :label-ecount (* batch-size
                      (ecount-from-map label-keys expected-ecount-map))
     :dataset
     (->> dataset
          (partition-all batch-size)
          (map
           (fn [dataset-batch]
             (when (not= batch-size (count dataset-batch))
               (throw (ex-info "Dataset size is not commensurate with batch size"
                               {:batch-size batch-size
                                :last-batch-count (count dataset-batch)})))
             (let [interleaved-items
                   (->> dataset-batch
                        (map
                         (fn [ds-entry]
                           (let [entry-values (check-entry-ecounts expected-ecount-map
                                                                   all-keys
                                                                   ds-entry
                                                                   options)
                                 leftover (apply dissoc ds-entry all-keys)]
                             (cond-> {::features (take n-features entry-values)
                                      ::label (drop n-features entry-values)}
                               (and keep-extra? (seq leftover))
                               (assoc :extra-data leftover))))))
                   leftover (->> (map :extra-data interleaved-items)
                                 (remove nil?)
                                 seq)]
               (cond-> {::features (mapcat ::features interleaved-items)
                        ::label (mapcat ::label interleaved-items)}
                 (and leftover keep-extra?)
                 (assoc :extra-data leftover))))))}))


(defn coalesce-dataset
  "Take a dataset and produce a sequence of values,label maps
  where the entries are coalesced items of the dataset.
  Ecounts are always checked.
options are:
  datatype - datatype to use.
  unchecked? - true for faster conversions to container.
  scalar-label? - true if the label should be a single scalar value.
  container-fn - container constructor with prototype:
     (container-fn datatype elem-count {:keys [unchecked?] :as options})
  queue-depth - parallelism used for coalescing - see tech.parallel/queued-pmap
    This is useful with the data sequence invoves cpu-intensive or blocking
    transformations (loading large images, scaling them, etc) and the train/test
    method is relatively fast in comparison.  Defaults to 0 in which case queued-pmap
    turns into just map.
  batch-size - nil - point by point conversion
             - number - items are coalesced into batches of given size.  Options map
                 passed in is passed to dataset->batched-dataset.
  keep-extra? - Keep extra data in the items.  Defaults to true.
                Allows users to assocthat users can assoc extra information into each
                data item for things like visualizations.

  Returns a sequence of
  {::features - container of datatype
   :labels - container or scalar}"
  [feature-keys label-keys {:keys [datatype
                                   unchecked?
                                   container-fn
                                   queue-depth
                                   batch-size
                                   keep-extra?
                                   ]
                            :or {datatype :float64
                                 unchecked? true
                                 container-fn dtype/make-array-of-type
                                 queue-depth 0
                                 batch-size 1}
                            :as options}
   dataset]
  (let [[dataset feature-keys label-keys
         value-ecount label-ecount]
        (let [{:keys [dataset value-ecount label-ecount]}
              (dataset->batched-dataset feature-keys label-keys batch-size
                                        options dataset)]
          [dataset [::features] (when (normalize-keys label-keys)
                                  [::label])
           value-ecount label-ecount])
        all-keys (concat feature-keys label-keys)
        n-features (count feature-keys)

        container-fn-options (merge options
                                    {:unchecked? unchecked?})
        value-ecount (long value-ecount)
        label-ecount (long label-ecount)]
    (->> dataset
         (parallel/queued-pmap
          queue-depth
          (fn [dataset-entry]
            ;;Batching takes care of label->integer(s) conversion so we do not do it
            ;;again
            (let [entry-data (dataset-entry->data all-keys dataset-entry
                                                  (dissoc options :label-map))

                  feature-data (take n-features entry-data)
                  label-data (seq (drop n-features entry-data))
                  feature-container (container-fn datatype value-ecount
                                                  container-fn-options)
                  label-container (when label-keys
                                    (container-fn datatype label-ecount
                                                  container-fn-options))]
              ;;Remove all used keys.  This saves potentially huge amounts of
              ;;memory.  That being said, there may be information on the dataset
              ;;entry that is useful to recreate sample so we are conservatively
              ;;keeping anything we didn't convert.
              (try
                (merge (if keep-extra?
                         (apply dissoc dataset-entry all-keys)
                         {})
                       {::features (first (dtype/copy-raw->item!
                                           feature-data feature-container 0
                                           container-fn-options))}
                       (when label-keys
                         {::label (if label-container
                                    (first (dtype/copy-raw->item!
                                            label-data label-container 0
                                            container-fn-options))
                                    (if unchecked?
                                      (let [label-val (-> (flatten label-data)
                                                          first)]
                                        (dtype/unchecked-cast label-val datatype)
                                        (dtype/cast label-val datatype))))}))
                ;;certain classes of errors are caught here.
                (catch Throwable e
                  (throw (ex-info "Failed to convert entry"
                                  {:error e
                                   :entry dataset-entry}))))))))))


(defn sequence->iterator
  "Java ml interfaces sometimes use iterators where they really should
  use sequences (iterators have state).  In any case, we do what we can."
  ^Iterator [item-seq]
  (let [next-item-fn (parallel/create-next-item-fn item-seq)
        next-item-atom (atom (next-item-fn))]
    (proxy [Iterator] []
      (hasNext []
        (boolean @next-item-atom))
      (next []
        (locking this
          (if-let [entry @next-item-atom]
            (do
              (reset! next-item-atom (next-item-fn))
              entry)
            (throw (NoSuchElementException.))))))))


(defn ->k-fold-datasets
  "Given 1 dataset, prepary K datasets using the k-fold algorithm.
  Randomize dataset defaults to true which will realize the entire dataset
  so use with care if you have large datasets."
  [k {:keys [randomize-dataset?]
      :or {randomize-dataset? true}}
   dataset]
  (let [dataset (cond-> dataset
                  randomize-dataset? shuffle)
        fold-size (inc (quot (count dataset) k))
        folds (vec (partition-all fold-size dataset))]
    (for [i (range k)]
      {:test-ds (nth folds i)
       :train-ds (apply concat (keep-indexed #(if (not= %1 i) %2) folds))})))


(defn ->train-test-split
  [{:keys [randomize-dataset? train-fraction]
    :or {randomize-dataset? true
         train-fraction 0.7}}
   dataset]
  (let [dataset (cond-> dataset
                  randomize-dataset? shuffle)
        n-elems (count dataset)
        n-training (long (Math/round (* n-elems (double train-fraction))))]
    {:train-ds (take n-training dataset)
     :test-ds (drop n-training dataset)}))


(defn- update-min-max
  [old-val batch-size new-val]
  ;;Ensure we have jvm-representable values (not pointers to c objects)
  (let [[min-container max-container]
        (or old-val
            (let [container-shape [(quot (dtype/ecount new-val)
                                         batch-size)]
                  first-row (ct/in-place-reshape new-val container-shape)]
              [(-> (ct/from-prototype new-val :shape container-shape)
                   (ct/assign! first-row))
               (-> (ct/from-prototype new-val :shape container-shape)
                   (ct/assign! first-row))]))]
    [(ops/min min-container new-val)
     (ops/max max-container new-val)]))


(defn per-parameter-dataset-min-max
  "Create a new (coalesced) dataset with parameters scaled.
If label range is not provided then labels are left unscaled."
  [batch-size coalesced-dataset]
  (let [batch-size (long (or batch-size 1))]
    (reduce (fn [min-max-map {:keys [::features ::label]}]
              (cond-> min-max-map
                features (update ::features update-min-max batch-size features)
                label (update ::label update-min-max batch-size label)))
            {}
            coalesced-dataset)))


(defn min-max-map->scale-map
  [min-max-map range-map]
  (->> min-max-map
       (map (fn [[k [min-v max-v]]]
              (if-let [range-data (get range-map k)]
                [k
                 (let [[min-val max-val] range-data
                       val-range (- (double max-val)
                                    (double min-val))
                       range-data (-> (ct/clone max-v)
                                      (ops/- min-v)
                                      (ops// val-range))]
                   {:per-elem-subtract min-v
                    :per-elem-div range-data
                    :per-elem-bias min-val})])))
       (into {})))


(defn per-parameter-scale-coalesced-dataset
  "scale a coalesced dataset in place"
  [scale-map coalesced-dataset]
  (->> coalesced-dataset
       (map
        (fn [ds-entry]
          (merge ds-entry
                 (->> scale-map
                      (map (fn [[k scale-entry]]
                             (when-let [v (get ds-entry k)]
                               [k (-> (ct/clone v)
                                      (ops/- (:per-elem-subtract scale-entry))
                                      (ops// (:per-elem-div scale-entry))
                                      (ops/+ (:per-elem-bias scale-entry)))])))
                      (remove nil?)
                      (into {})))))))


(defn post-process-coalesced-dataset
  [options feature-keys key-ecount-map label-keys coalesced-dataset]
  (let [label-map (:label-map options)
        options (merge options
                       {::dataset-info
                        (merge {::feature-ecount (->> feature-keys
                                                      (map key-ecount-map)
                                                      (apply +))
                                ::key-ecount-map key-ecount-map}
                               (when (and (= 1 (count label-keys))
                                          (get label-map (first label-keys)))
                                 {::num-classes (count (get label-map
                                                            (first label-keys)))}))
                        ::feature-keys feature-keys
                        ::label-keys label-keys})]
    (cond
      (:range-map options)
      (let [min-max-map (per-parameter-dataset-min-max (:batch-size options)
                                                       coalesced-dataset)
            scale-map (min-max-map->scale-map min-max-map (:range-map options))]
        {:coalesced-dataset (per-parameter-scale-coalesced-dataset
                             scale-map coalesced-dataset)
         :options (-> (dissoc options :range-map)
                      (assoc :scale-map scale-map))})
      (:scale-map options)
      (let [scale-map (:scale-map options)]
        {:coalesced-dataset (per-parameter-scale-coalesced-dataset
                             scale-map coalesced-dataset)
         :options options})
      :else
      {:coalesced-dataset coalesced-dataset
       :options options})))


(defn apply-dataset-options
  "Apply dataset options to dataset producing a coalesced dataset and a new options map.
  A coalesced dataset is a dataset where all the feature keys are coalesced into a
  contiguous ::features member and all the labels are coalesced into a contiguous ::labels
  member.

  Transformations:

  If the dataset as nominal (not numeric) data then this data is converted into integer
  data and the original keys mapped to the indexes.  This is recorded in :label-map.

  Some global information about the dataset is recorded:
  ::dataset-info {:value-ecount - Ecount of the feature vector.
                 :key-ecount-map - map of keys to ecounts for all keys.}

  :feature-keys normaliaed feature keys.
  :label-keys normalized label keys.

  :range-map - if passed in, coalesced ::features or ::label's are set to the ranges
  specified in the map.  This means a min-max pass is performed and per-element scaling
  is done.  See tests for example. The result of a range map operation is a per-element
  scale map.

  :scale-map - if passed in, this is a map of #{::features ::label} to a scaling operation:
       (-> (ct/clone v)
           (ops/- (:per-elem-subtract scale-entry))
           (ops// (:per-elem-div scale-entry))
           (ops/+ (:per-elem-bias scale-entry)))

  "
  [feature-keys label-keys options dataset]
  (let [first-item (first dataset)]
    (if (and (contains? first-item ::features)
             (contains? first-item ::label)
             (contains? options ::dataset-info))
      ;;This has already been coalesced
      (do
        (when-not (and (= [::features] (normalize-keys feature-keys))
                       (or (= [::label] (normalize-keys label-keys))
                           (= nil label-keys)))
          (throw (ex-info "Dataset appears coalesced but new keys appear to be added"
                          {:feature-keys feature-keys
                           :label-keys label-keys})))
        {:options options
         :coalesced-dataset
         ;;Then we re-coalesce as this may change the container type.
         (coalesce-dataset feature-keys label-keys options dataset)})
      ;;Else do the work of analyzing and coalescing the dataset.
      (let [feature-keys (normalize-keys feature-keys)
            label-keys (normalize-keys label-keys)
            all-keys (concat feature-keys label-keys)
            key-ecount-map (->> (dataset-entry->data all-keys first-item {})
                                (ecount-map all-keys))
            categorical-labels (->> all-keys
                                    (map #(let [item (get-dataset-item
                                                      first-item % {})]
                                            (when (or (string? item)
                                                      (keyword? item))
                                              %)))
                                    (remove nil?)
                                    seq)
            label-map (get options :label-map)
            label-map
            ;;scan dataset for labels.
            (if (and categorical-labels
                     (not label-map))
              (let [label-atom (atom {})
                    label-base-idx (long (or (:multiclass-label-base-index options)
                                             0))
                    map-fn (if (:deterministic-label-map? options)
                             map
                             pmap)]
                (->> dataset
                     (map-fn
                      (fn [ds-entry]
                        (->> categorical-labels
                             (map (fn [cat-label]
                                    (let [ds-value (get ds-entry cat-label)]
                                      (swap! label-atom update cat-label
                                             (fn [existing]
                                               (if-let [idx (get existing ds-value)]
                                                 existing
                                                 (assoc existing ds-value
                                                        (+ label-base-idx
                                                           (count existing)))))))))
                             dorun)
                        ds-entry))
                     dorun)
                @label-atom))
            options (merge options
                           (when label-map
                             {:label-map label-map}))]
        (->> (coalesce-dataset feature-keys label-keys
                               options dataset)
             (post-process-coalesced-dataset
              options feature-keys key-ecount-map label-keys))))))


(defn check-dataset-datatypes
  "Check that the datatype of the rest of the dataset matches
  the datatypes of the first entry."
  [dataset]
  (let [ds-types (->> (first dataset)
                      (map (fn [[k v]]
                             [k (number? v)]))
                      (into {}))]
    (->> dataset
         (mapcat (fn [ds-entry]
                   (->> ds-entry
                        (remove (fn [[dk dv]]
                                  (if (ds-types dk)
                                    (number? dv)
                                    (not (number? dv)))))
                        seq)))
         (remove nil?)
         set)))


(defn force-keyword
  "Force a value to a keyword.  Often times data is backwards where normative
  values are represented by numbers; this removes important information from
  a dataset.  If a particular column is categorical, it should be represented
  by a keyword."
  [value & {:keys [missing-value-placeholder]
            :or {missing-value-placeholder -1}}]
  (cond
    (number? value)
    (if (= value missing-value-placeholder)
      :unknown
      (keyword (str (long value))))
    (string? value)
    (keyword value)
    :else
    value))


(defn calculate-nominal-stats
  "Calculate the mean, variance of each value of each nominal-type feature
  as it relates to the regressed value.  This is useful to provide a simple
  number of derived features that directly relate to regressed values and
  that often provide better learning."
  [nominal-feature-keywords label-keyword dataset]
  (let [label-seq (map #(get % label-keyword) dataset)
        keyword-stats
        (->> nominal-feature-keywords
             (pmap
              (fn [kwd]
                [kwd (->> dataset
                          (group-by kwd)
                          (map (fn [[item-key item-val-seq]]
                                 [item-key
                                  {:mean (m-stats/mean (map label-keyword
                                                            item-val-seq))
                                   :variance (if (> (count item-val-seq)
                                                    1)
                                               (m-stats/variance (map label-keyword
                                                                      item-val-seq))
                                               0)}]))
                          (into {}))]))
             (into {}))]
    (assoc keyword-stats
           label-keyword {:mean (m-stats/mean label-seq)
                          :variance (m-stats/variance label-seq)})))


(defn augment-dataset-with-stats
  [stats-map nominal-feature-keywords label-keyword dataset]
  (->>
   dataset
   (map (fn [ds-entry]
          (reduce
           (fn [ds-entry feature-key]
             (let [key-mean (keyword (str (name feature-key) "mean"))
                   key-var (keyword (str (name feature-key) "variance"))]
               (if-let [stats-entry (get-in stats-map
                                            [feature-key
                                             (get ds-entry feature-key)])]
                 (utils/prefix-merge (name feature-key) ds-entry stats-entry)
                 (utils/prefix-merge (name feature-key) ds-entry
                                     (get stats-map label-keyword)))))
           ds-entry
           nominal-feature-keywords)))))

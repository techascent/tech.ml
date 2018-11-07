(ns tech.ml.dataset
  "The most simple dataset description we have figured out is a sequence of maps.

  Using this definition, things like k-fold have natural interpretations.

  While this works well for clojure generation/manipulation,  ML interfaces uniformly
  require this sequence to be coalesced somehow into larger buffers; sometimes on a
  point by point basis and sometimes into batching buffers.  This file is intended
  to provide direct, simple tools to provide either type of coalesced information.

  Care has been taken to keep certain operations lazy so that datasets of unbounded
  length can be manipulated."
  (:require [tech.datatype :as dtype]
            [tech.parallel :as parallel]
            [tech.compute.cpu.tensor-math :as cpu-tm]
            [tech.compute.tensor :as ct]
            [tech.compute.tensor.operations :as ops])
  (:import [java.util Iterator NoSuchElementException]))


(set! *warn-on-reflection* true)


(defn get-dataset-item
  [dataset-entry item-key]
  (if-let [retval (get dataset-entry item-key)]
    retval
    (throw (ex-info "Failed to get dataset item"
                    {:item-key item-key
                     :dataset-entry-keys (keys dataset-entry)}))))


(defn- ecount-map
  [entry-keys dataset-entry-values]
  (->> (map dtype/ecount dataset-entry-values)
       (zipmap entry-keys)))


(defn- dataset-entry->data
  [entry-keys dataset-entry]
  (->> entry-keys
       (map #(get-dataset-item dataset-entry %))))


(defn- normalize-keys
  [kwd-or-seq]
  (when kwd-or-seq
    (-> (if-not (sequential? kwd-or-seq)
          [kwd-or-seq]
          kwd-or-seq)
        seq)))


(defn- check-entry-ecounts
  [expected-ecount-map all-keys ds-entry]
  (let [entry-values (dataset-entry->data all-keys ds-entry)
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
  {:values :label :extra-data}"
  [feature-keys label-keys batch-size {:keys [keep-extra?]
                                       :or {keep-extra? true}}
   dataset]
  (let [feature-keys (normalize-keys feature-keys)
        label-keys (normalize-keys label-keys)
        n-features (count feature-keys)
        all-keys (concat feature-keys label-keys)
        expected-ecount-map (->> (dataset-entry->data all-keys (first dataset))
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
                                                                   ds-entry)
                                 leftover (apply dissoc ds-entry all-keys)]
                             (cond-> {:values (take n-features entry-values)
                                      :label (drop n-features entry-values)}
                               (and keep-extra? (seq leftover))
                               (assoc :extra-data leftover))))))
                   leftover (->> (map :extra-data interleaved-items)
                                 (remove nil?)
                                 seq)]
               (cond-> {:values (mapcat :values interleaved-items)
                        :label (mapcat :label interleaved-items)}
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
  {:values - container of datatype
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
  ;;Quick out of this dataset has already been coalesced
  dataset
  (let [[dataset feature-keys label-keys
         value-ecount label-ecount
         expected-ecount-map]
        (let [{:keys [dataset value-ecount label-ecount]}
              (dataset->batched-dataset feature-keys label-keys batch-size
                                        options dataset)]
          [dataset [:values] (when (normalize-keys label-keys)
                               [:label])
           value-ecount label-ecount nil])
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
            (let [entry-data (dataset-entry->data all-keys dataset-entry)
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
              (merge (if keep-extra?
                       (apply dissoc dataset-entry all-keys)
                       {})
                     {:values (first (dtype/copy-raw->item!
                                      feature-data feature-container 0
                                      container-fn-options))}
                     (when label-keys
                       {:label (if label-container
                                 (first (dtype/copy-raw->item!
                                         label-data label-container 0
                                         container-fn-options))
                                 (if unchecked?
                                   (let [label-val (-> (flatten label-data)
                                                       first)]
                                     (dtype/unchecked-cast label-val datatype)
                                     (dtype/cast label-val datatype))))}))))))))


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


(defn dataset->k-fold-datasets
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


(defn- update-min-max
  [old-val new-val]
  ;;Ensure we have jvm-representable values (not pointers to c objects)
  (if-not old-val
    (let [min-container (ct/clone new-val)
          max-container (ct/clone new-val)]
      [min-container max-container])
    (let [[min-container max-container] old-val]
      [(ops/min min-container new-val)
       (ops/max max-container new-val)])))


(defn per-parameter-dataset-min-max
  "Create a new (coalesced) dataset with parameters scaled.
If label range is not provided then labels are left unscaled."
  [coalesced-dataset]
  (reduce (fn [min-max-map {:keys [values label]}]
            (cond-> min-max-map
              values (update :values update-min-max values)
              label (update :label update-min-max label)))
          {}
          coalesced-dataset))


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


(defn per-parameter-scale-coalesced-dataset!
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


(defn apply-dataset-options
  [feature-keys label-keys options dataset]
  (let [coalesced-dataset (coalesce-dataset feature-keys label-keys
                                            options dataset)]
    (cond
      (:range-map options)
      (let [min-max-map (per-parameter-dataset-min-max coalesced-dataset)
            scale-map (min-max-map->scale-map min-max-map (:range-map options))]
        {:coalesced-dataset (per-parameter-scale-coalesced-dataset!
                             scale-map coalesced-dataset)
         :options (-> (dissoc options :range-map)
                      (assoc :scale-map scale-map))})
      (:scale-map options)
      (let [scale-map (:scale-map options)]
        {:coalesced-dataset (per-parameter-scale-coalesced-dataset!
                             scale-map coalesced-dataset)
         :options options})
      :else
      {:coalesced-dataset coalesced-dataset
       :options options})))

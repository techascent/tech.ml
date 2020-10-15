(ns tech.libs.xgboost
  (:require [tech.v2.datatype :as dtype]
            [tech.ml.model :as model]
            [tech.ml.protocols.system :as ml-proto]
            [tech.ml.registry :as ml-registry]
            [tech.ml.dataset.utils :as utils]
            [tech.ml.util :as ml-util]
            [tech.ml.gridsearch :as ml-gs]
            [clojure.string :as s]
            [clojure.set :as set]
            [tech.ml.dataset :as ds]
            [clojure.tools.logging :as log])
  (:refer-clojure :exclude [load])
  (:import [ml.dmlc.xgboost4j.java Booster XGBoost XGBoostError DMatrix]
           [ml.dmlc.xgboost4j LabeledPoint]
           [java.util Iterator UUID LinkedHashMap Map]
           [java.io ByteArrayInputStream ByteArrayOutputStream]))


(set! *warn-on-reflection* true)


(defmulti model-type->xgboost-objective
  (fn [model-type]
    model-type))


(def objective-types
  {:linear-regression "reg:linear"
   :squared-error-regression "reg:squarederror"
   :logistic-regression "reg:logistic"
   ;;logistic regression for binary classification
   :logistic-binary-classification "binary:logistic"
   ;; logistic regression for binary classification, output score before logistic
   ;; transformation
   :logistic-binary-raw-classification "binary:logitraw"
    ;;hinge loss for binary classification. This makes predictions of 0 or 1, rather
    ;;than producing probabilities.
   :binary-hinge-loss "binary:hinge"
   ;; versions of the corresponding objective functions evaluated on the GPU; note that
   ;; like the GPU histogram algorithm, they can only be used when the entire training
   ;; session uses the same dataset
   :gpu-linear-regression "gpu:reg:linear"
   :gpu-logistic-regression "gpu:reg:logistic"
   :gpu-binary-logistic-classification "gpu:binary:logistic"
   :gpu-binary-logistic-raw-classification "gpu:binary:logitraw"

   ;; poisson regression for count data, output mean of poisson distribution
   ;; max_delta_step is set to 0.7 by default in poisson regression (used to safeguard
   ;; optimization)
   :count-poisson "count:poisson"

   ;; Cox regression for right censored survival time data (negative values are
   ;; considered right censored). Note that predictions are returned on the hazard ratio
   ;; scale (i.e., as HR = exp(marginal_prediction) in the proportional hazard function
   ;; h(t) = h0(t) * HR).
   :survival-cox "survival:cox"
   ;; set XGBoost to do multiclass classification using the softmax objective, you also
   ;; need to set num_class(number of classes)
   :multiclass-softmax "multi:softmax"
   ;; same as softmax, but output a vector of ndata * nclass, which can be further
   ;; reshaped to ndata * nclass matrix. The result contains predicted probability of
   ;; each data point belonging to each class.
   :multiclass-softprob "multi:softprob"
   ;; Use LambdaMART to perform pairwise ranking where the pairwise loss is minimized
   :rank-pairwise "rank:pairwise"
   ;; Use LambdaMART to perform list-wise ranking where Normalized Discounted Cumulative
   ;; Gain (NDCG) is maximized
   :rank-ndcg "rank:ndcg"
   ;; Use LambdaMART to perform list-wise ranking where Mean Average Precision (MAP) is
   ;; maximized
   :rank-map "rank:map"
   ;; gamma regression with log-link. Output is a mean of gamma distribution. It might
   ;; be useful, e.g., for modeling insurance claims severity, or for any outcome that
   ;; might be gamma-distributed.
   :gamma-regression "reg:gamma"
    ;; Tweedie regression with log-link. It might be useful, e.g., for modeling total
    ;; loss in insurance, or for any outcome that might be Tweedie-distributed.
   :tweedie-regression "reg:tweedie"})


(defmethod model-type->xgboost-objective :default
  [model-type]
  (if-let [retval (get objective-types model-type)]
    retval
    (throw (ex-info "Unrecognized xgboost model type"
                    {:model-type model-type
                     :possible-types (keys objective-types)}))))


(defmethod model-type->xgboost-objective :regression
  [_]
  (model-type->xgboost-objective :squared-error-regression))


(defmethod model-type->xgboost-objective :binary-classification
  [_]
  (model-type->xgboost-objective :logistic-binary-classification))


(defmethod model-type->xgboost-objective :classification
  [_]
  (model-type->xgboost-objective :multiclass-softprob))


;; For a full list of options, check out:
;; https://xgboost.readthedocs.io/en/latest/parameter.html

(defn load
  ^Booster [^String path]
  (XGBoost/loadModel path))


(defn save
  [^Booster trained-model ^String path]
  (.saveModel trained-model path))


(defn model->byte-array
  ^bytes [^Booster model]
  (model/model->byte-array model))


(defn byte-array->model
  ^Booster [^bytes data]
  (model/byte-array->model data))


(defn- ->data
  [item default-value]
  (if item
    (dtype/get-value item 0)
    default-value))


(defn dataset->labeled-point-iterator
  "Create an iterator to labeled points from a possibly quite large
  sequence of maps.  Sets expected length to length of first entry"
  ^Iterator [dataset options]
  (->> (ds/->row-major dataset (assoc options :datatype :float32))
       ;;dataset is now coalesced into float arrays for the values
       ;;and a single float for the label (if it exists).
       (map (fn [{:keys [:features :label]}]
              (LabeledPoint. (float (->data label 0.0)) nil ^floats features)))
       (utils/sequence->iterator)))


(defn dataset->dmatrix
  "Dataset is a sequence of maps.  Each contains a feature key.
  Returns a dmatrix."
  ^DMatrix [dataset options]
  (DMatrix. (dataset->labeled-point-iterator dataset options)
            nil))


(defn- get-objective
  [options]
  (model-type->xgboost-objective
   (or (model/options->model-type options)
       :linear-regression)))

(defn multiclass-objective?
  [objective]
  (or (= objective "multi:softmax")
      (= objective "multi:softprob")))


(defn- safe-name
  [item]
  (if (or (keyword? item)
          (symbol? item))
    (name item)
    (str item)))


(defn def-ml-model
  [model-keyword train-fn predict-fn {:keys [hyper-parameter-map
                                             thaw-fn
                                             explain-fn]}]

  )


(defrecord XGBoostSystem []
  ml-proto/PMLSystem
  (system-name [_] :xgboost)
  (gridsearch-options [system options]
    ;;These are just a very small set of possible options:
    ;;https://xgboost.readthedocs.io/en/latest/parameter.html
    {:subsample {:doc "blbalhbh"
                 :gridsearch (ml-gs/linear [0.1 1.0])}
     :scale-pos-weight (ml-gs/linear [0.1 2.0])
     :max-depth (comp long (ml-gs/linear-long [2 50]))
     :lambda (ml-gs/linear [0.01 2])
     :gamma (ml-gs/exp [0.001 100])
     :eta (ml-gs/linear [0 1])
     :alpha (ml-gs/exp [0.01 2])})

  (train [_ options dataset]
    ;;XGBOOST is fully reentrant but it doesn't benefit from further explicit
    ;;parallelization.  Because of this, allowing xgboost to be 'pmapped'
    ;;results a lot of times in a heavily oversubscribed machine.
    (locking _
      (let [objective (get-objective options)
            train-dmat (dataset->dmatrix dataset options)
            base-watches (or (:watches options) {})
            watches (->> base-watches
                         (reduce (fn [^Map watches [k v]]
                                   (.put watches (safe-name k)
                                         (dataset->dmatrix v options))
                                   watches)
                                 ;;Linked hash map to preserve order
                                 (LinkedHashMap.)))
            round (or (:round options) 25)
            early-stopping-round (or (when (:early-stopping-round options)
                                       (int (:early-stopping-round options)))
                                     0)
            _ (when (and (> (count watches) 1)
                         (not (instance? LinkedHashMap (:watches options)))
                         (not= 0 early-stopping-round))
                (log/warn "Early stopping indicated but watches has undefined iteration order.
Early stopping will always use the 'last' of the watches as defined by the iteration
order of the watches map.  Consider using a java.util.LinkedHashMap for watches.
https://github.com/dmlc/xgboost/blob/master/jvm-packages/xgboost4j/src/main/java/ml/dml
c/xgboost4j/java/XGBoost.java#L208"))
            watch-names (->> base-watches
                             (map-indexed (fn [idx [k v]]
                                            [idx k]))
                             (into {}))
            label-map (when (multiclass-objective? objective)
                        (ds/inference-target-label-map dataset))
            params (->> (-> (dissoc options :model-type :watches)
                            (assoc :objective objective))
                        ;;Adding in some defaults
                        (merge {}
                               {
                                :alpha 0.0
                                :eta 0.3
                                :lambda 1.0
                                :max-depth 6
                                :scale-pos-weight 1.0
                                :subsample 0.87
                                :silent 1
                                }
                               options
                               (when label-map
                                 {:num-class (count label-map)}))
                        (map (fn [[k v]]
                               (when v
                                 [(s/replace (name k) "-" "_" ) v])))
                        (remove nil?)
                        (into {}))
            ^"[[F" metrics-data (when-not (empty? watches)
                                  (->> (repeatedly (count watches)
                                                   #(float-array round))
                                       (into-array)))
            ^Booster model (XGBoost/train train-dmat params
                                          (long round)
                                          (or watches {}) metrics-data nil nil
                                          (int early-stopping-round))
            out-s (ByteArrayOutputStream.)]
        (.saveModel model out-s)
        {:model-data (.toByteArray out-s)
         :metrics
         (->> watches
              (map-indexed vector)
              (map (fn [[watch-idx [watch-name watch-data]]]
                     [(get watch-names watch-idx)
                      (aget metrics-data watch-idx)]))
              (ds/name-values-seq->dataset)
              (#(vary-meta % assoc :name :metrics)))})))

  (thaw-model [_ model]
    (-> (if (map? model)
          (:model-data model)
          model)
        (ByteArrayInputStream.)
        (XGBoost/loadModel)))

  (predict [_ options thawed-model dataset]
    (let [retval (->> (dataset->dmatrix dataset (dissoc options :label-columns))
                      (.predict ^Booster thawed-model))]
      (if (= "multi:softprob" (get-objective options))
        (dtype/make-reader
         :posterior-probabilities
         (alength retval)
         (aget retval idx))
        (dtype/make-reader
         :float32
         (alength retval)
         (aget ^floats (aget retval idx) 0)))))

  ml-proto/PMLExplain
  ;;"https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7"
  (explain-model [this model {:keys [importance-type feature-columns]
                           :or {importance-type "gain"}}]
    (let [^Booster booster (ml-proto/thaw-model this model)
          feature-col-map (->> feature-columns
                               (map (fn [name]
                                      [name (ml-util/->str name)]))
                               (into {}))
          feature-columns (into-array String (map #(get feature-col-map %)
                                                  feature-columns))
          ^Map score-map (.getScore booster
                                    ^"[Ljava.lang.String;" feature-columns
                                    ^String importance-type)
          col-inv-map (set/map-invert feature-col-map)]
      ;;It's not a great map...Something is off about iteration so I have
      ;;to transform it back into something sane.
      (->> (keys score-map)
           (map (fn [item-name]
                  {:importance-type importance-type
                   :colname (get col-inv-map item-name)
                   (keyword importance-type) (.get score-map item-name)}))
           (sort-by (keyword importance-type) >)
           (ds/->>dataset)))))


(def system
  (memoize
   (fn []
     (->XGBoostSystem))))


(ml-registry/register-system (system))

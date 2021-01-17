(ns tech.v3.libs.xgboost
  "Require this namespace to get xgboost support for classification and regression.
  Defines a full range of xgboost model definitions and supports xgboost explain
  functionality."
  (:require [tech.v3.datatype :as dtype]
            [tech.v3.datatype.errors :as errors]
            [tech.v3.ml :as ml]
            [tech.v3.ml.model :as model]
            [tech.v3.ml.gridsearch :as ml-gs]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.tensor :as ds-tens]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.dataset.utils :as ds-utils]
            [tech.v3.tensor :as dtt]
            [clojure.set :as set]
            [clojure.string :as s]
            [clojure.tools.logging :as log])
  (:import [ml.dmlc.xgboost4j.java Booster XGBoost XGBoostError DMatrix]
           [ml.dmlc.xgboost4j LabeledPoint]
           [java.util Iterator UUID LinkedHashMap Map]
           [java.io ByteArrayInputStream ByteArrayOutputStream]))


(set! *warn-on-reflection* true)


(def ^:private objective-types
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


(defmulti ^:private model-type->xgboost-objective
  (fn [model-type]
    model-type))


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



(defn- sparse->labeled-point [sparse target n-sparse-columns]
  (let [x-i-s
        (map
         #(hash-map :i  (.i %) :x (.x %))
         (iterator-seq
          (.iterator sparse)))]
    (LabeledPoint. target
                   n-sparse-columns
                   (into-array Integer/TYPE (map :i x-i-s))
                   (into-array Float/TYPE (map :x x-i-s)))))

(defn- sparse-feature->dmatrix [feature-ds target-ds sparse-column n-sparse-columns]
  (DMatrix.
   (.iterator
    (map
     (fn [features target ] (sparse->labeled-point features target n-sparse-columns))
     (get feature-ds sparse-column)
     (or  (get target-ds (first (ds-mod/inference-target-column-names target-ds)))
          (repeat 0.0)
          ))) nil))



(defn- dataset->labeled-point-iterator
  "Create an iterator to labeled points from a possibly quite large
  sequence of maps.  Sets expected length to length of first entry"
  ^Iterable [feature-ds target-ds]
  (let [feature-tens (ds-tens/dataset->tensor feature-ds :float32)
        target-tens (when target-ds
                      (ds-tens/dataset->tensor target-ds :float32))]
    (errors/when-not-errorf
     (or (not target-ds)
         (== 1 (ds/column-count target-ds)))
     "Multi-column regression/classification is not supported.  Target ds has %d columns"
     (ds/column-count target-ds))
    (map (fn [features target]
           (LabeledPoint. (float target) (first (dtype/shape features))  nil (dtype/->float-array features)))
         feature-tens (or (when target-tens (dtype/->reader target-tens))
                          (repeat (float 0.0))))))

(defn- dataset->dmatrix
  "Dataset is a sequence of maps.  Each contains a feature key.
  Returns a dmatrix."
  (^DMatrix [feature-ds target-ds]
   (DMatrix. (.iterator (dataset->labeled-point-iterator feature-ds target-ds))
             nil))
  (^DMatrix [feature-ds]
   (dataset->dmatrix feature-ds nil)))


(defn- options->objective
  [options]
  (model-type->xgboost-objective
   (or (when (:model-type options)
         (keyword (name (:model-type options))))
       :linear-regression)))


(defn- multiclass-objective?
  [objective]
  (or (= objective "multi:softmax")
      (= objective "multi:softprob")))


(def ^:private hyperparameters
  {:subsample (ml-gs/linear 0.7 1.0 3)
   :scale-pos-weight (ml-gs/linear 0.7 1.31 6)
   :max-depth (ml-gs/linear 1 10 10 :int64)
   :lambda (ml-gs/linear 0.01 0.31 30)
   :gamma (ml-gs/linear 0.001 1 10)
   :eta (ml-gs/linear 0 1 10)
   :round (ml-gs/linear 5 46 5 :int64)
   :alpha (ml-gs/linear 0.01 0.31 30)})

(defn ->dmatrix [feature-ds target-ds sparse-column n-sparse-columns]
  (if sparse-column
    (sparse-feature->dmatrix feature-ds target-ds sparse-column n-sparse-columns)
    (dataset->dmatrix feature-ds target-ds)))

(defn- train
  [feature-ds label-ds options]
  ;;XGBoost uses all cores so serialization here avoids over subscribing
  ;;the machine.
  (locking #'multiclass-objective?
    (let [objective (options->objective options)
          sparse-column-or-nil (:sparse-column options)
          train-dmat (->dmatrix feature-ds label-ds sparse-column-or-nil (:n-sparse-columns options))
          base-watches (or (:watches options) {})
          feature-cnames (ds/column-names feature-ds)
          target-cnames (ds/column-names label-ds)
          watches (->> base-watches
                       (reduce (fn [^Map watches [k v]]
                                 (.put watches (ds-utils/column-safe-name k)
                                       (->dmatrix
                                        (ds/select-columns v feature-cnames)
                                        (ds/select-columns v target-cnames)
                                        sparse-column-or-nil
                                        (:n-sparse-columns options)))
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
                      (ds-mod/inference-target-label-map label-ds))
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
      (merge
       {:model-data (.toByteArray out-s)}
       (when (seq watches)
         {:metrics
          (->> watches
               (map-indexed vector)
               (map (fn [[watch-idx [watch-name watch-data]]]
                      [(get watch-names watch-idx)
                       (aget metrics-data watch-idx)]))
               (into {})
               (ds/->>dataset {:dataset-name :metrics}))})))))


(defn- thaw-model
  [model-data]
  (-> (if (map? model-data)
        (:model-data model-data)
        model-data)
      (ByteArrayInputStream.)
      (XGBoost/loadModel)))


(defn- predict
  [feature-ds thawed-model {:keys [target-columns target-categorical-maps options]}]
  (let [sparse-column-or-nil (:sparse-column options)
        dmatrix (->dmatrix feature-ds nil sparse-column-or-nil (:n-sparse-columns options))
        prediction (.predict ^Booster thawed-model dmatrix)
        predict-ds (->> prediction
                        (dtt/->tensor))
        target-cname (first target-columns)]
    (if (multiclass-objective? (options->objective options))
      (model/finalize-classification predict-ds
                                     (ds/row-count feature-ds)
                                     target-cname
                                     target-categorical-maps)
      (model/finalize-regression predict-ds target-cname))))


(defn- explain
  [thawed-model {:keys [feature-columns options]}
   {:keys [importance-type]
    :or {importance-type "gain"}}]
  (let [^Booster booster thawed-model
        sparse-column-or-nil (:sparse-column options)]
    (if sparse-column-or-nil
      (let [score-map (.getScore booster "" importance-type)]
        (ds/->dataset {:feature (keys score-map)
                       (keyword importance-type) (vals score-map) }))
      (let [feature-col-map (->> feature-columns
                                 (map (fn [name]
                                        [name (ds-utils/column-safe-name name)]))
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
             (ds/->>dataset))))))


(doseq [objective (concat [:regression :classification]
                          (keys objective-types))]
  (ml/define-model! (keyword "xgboost" (name objective))
    train predict {:thaw-fn thaw-model
                   :explain-fn explain
                   :hyperparameters hyperparameters}))


(comment
  (require '[tech.v3.dataset.column-filters :as cf])
  (def src-ds (ds/->dataset "test/data/iris.csv"))
  (def ds (->  src-ds
               (ds/categorical->number cf/categorical)
               (ds-mod/set-inference-target "species")))
  (def feature-ds (cf/feature ds))
  (def split-data (ds-mod/train-test-split ds))
  (def train-ds (:train-ds split-data))
  (def test-ds (:test-ds split-data))
  (def model (ml/train train-ds {:validate-parameters 1
                                 :round 10
                                 :silent 0
                                 :verbosity 3
                                 :model-type :xgboost/classification}))
  (def predictions (ml/predict test-ds model))
  (ml/explain model)
  (require '[tech.v3.ml.loss :as loss])
  (require '[tech.v3.dataset.categorical :as ds-cat])

  (loss/classification-accuracy (predictions "species")
                                (test-ds "species"))
  ;;0.93333

  (def titanic (-> (ds/->dataset "test/data/titanic.csv")
                   (ds/drop-columns ["Name"])
                   (ds/update-column "Survived" (fn [col]
                                                  (dtype/emap #(if (== 1 (long %))
                                                                 "survived"
                                                                 "drowned")
                                                              :string col)))
                   (ds-mod/set-inference-target "Survived")))

  (def titanic-numbers (ds/categorical->number titanic cf/categorical))

  (def split-data (ds-mod/train-test-split titanic-numbers))
  (def train-ds (:train-ds split-data))
  (def test-ds (:test-ds split-data))
  (def model (ml/train train-ds {:model-type :xgboost/classification}))
  (def predictions (ml/predict test-ds model))

  (loss/classification-accuracy (predictions "Survived")
                                (test-ds "Survived"))
  ;;0.8195488721804511
  ;;0.8308270676691729
  (require '[tech.v3.ml.gridsearch :as ml-gs])
  (def opt-map (merge {:model-type :xgboost/classification}
                      hyperparameters))
  (def options-sequence (take 200  (ml-gs/sobol-gridsearch opt-map)))


  (defn test-options
    [options]
    (let [model (ml/train train-ds options)
          predictions (ml/predict test-ds model)
          loss (loss/classification-loss (predictions "Survived")
                                         (test-ds "Survived"))]
      (assoc model :loss loss)))


  (def models
    (->> (map test-options options-sequence)
         (sort-by :loss)
         (take 10)
         (map #(select-keys % [:loss :options]))))
  ;;consistently gets .849 or so accuracy on best models.
  )

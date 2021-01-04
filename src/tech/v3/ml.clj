(ns tech.v3.ml
  "Simple machine learning based on tech.v3.dataset functionality."
  (:require [tech.v3.datatype.errors :as errors]
            [tech.v3.datatype.argops :as argops]
            [tech.v3.datatype.functional :as dfn]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod]
            [tech.v3.ml.loss :as loss]
            [tech.v3.ml.gridsearch :as ml-gs]
            [clojure.tools.logging :as log])
  (:import [java.util UUID]))


(defonce ^{:doc "Map of model kwd to model definition"} model-definitions* (atom nil))


(defn define-model!
  "Create a model definition.  An ml model is a function that takes a dataset and an
  options map and returns a model.  A model is something that, combined with a dataset,
  produces a inferred dataset."
  [model-kwd train-fn predict-fn {:keys [hyperparameters
                                         thaw-fn
                                         explain-fn
]}]
  (swap! model-definitions* assoc model-kwd {:train-fn train-fn
                                             :predict-fn predict-fn
                                             :hyperparameters hyperparameters
                                             :thaw-fn thaw-fn
                                             :explain-fn explain-fn

                                             })
  :ok)

(defn model-definition-names
  "Return a list of all registered model defintion names."
  []
  (keys @model-definitions*))


(defn options->model-def
  "Return the model definition that corresponse to the :model-type option"
  [options]
  (if-let [model-def (get @model-definitions* (:model-type options))]
    model-def
    (errors/throwf "Failed to find model %s.  Is a require missing?" (:model-type options))))


(defn hyperparameters
  "Get the hyperparameters for this model definition"
  [model-kwd]
  (:hyperparameters (options->model-def {:model-type model-kwd})))


(defn preprocess [dataset options]
  (let [fn (get-in options [:preprocess :fn] identity)]
    (fn dataset (:preprocess options))
    )

  )

(defn train
  "Given a dataset and an options map produce a model.  The model-type keyword in the
  options map selects which model definition to use to train the model.  Returns a map
  containing at least:


  * `:model-data` - the result of that definitions's train-fn.
  * `:options` - the options passed in.
  * `:id` - new randomly generated UUID.
  * `:feature-columns - vector of column names.
  * `:target-columns - vector of column names."
  [dataset options]
  (let [{:keys [train-fn]} (options->model-def options)
        feature-ds (cf/feature dataset)
        _ (errors/when-not-error (> (ds/row-count feature-ds) 0)
                                 "No features provided")
        target-ds (cf/target dataset)
        _ (errors/when-not-error (> (ds/row-count target-ds) 0)
                                 "No target columns provided
see tech.v3.dataset.modelling/set-inference-target")
        model-data (train-fn feature-ds target-ds options)
        cat-maps (ds-mod/dataset->categorical-xforms target-ds)]
    (merge
     {:model-data model-data
      :options options
      :id (UUID/randomUUID)
      :feature-columns (vec (ds/column-names feature-ds))
      :target-columns (vec (ds/column-names target-ds))}
     (when-not (== 0 (count cat-maps))
       {:target-categorical-maps cat-maps}))))


(defn thaw-model
  "Thaw a model.  Model's returned from train may be 'frozen' meaning a 'thaw'
  operation is needed in order to use the model.  This happens for you during preduct
  but you may also cached the 'thawed' model on the model map under the
  ':thawed-model'  keyword in order to do fast predictions on small datasets."
  [model {:keys [thaw-fn]}]
  (if-let [cached-model (get model :thawed-model)]
    cached-model
    (if thaw-fn
      (thaw-fn (get model :model-data))
      (get model :model-data))))


(defn predict
  "Predict returns a dataset with only the predictions in it.

  * For regression, a single column dataset is returned with the column named after the
    target
  * For classification, a dataset is returned with a float64 column for each target
    value and values that describe the probability distribution."
  [dataset model]
  (let [{:keys [predict-fn] :as model-def} (options->model-def (:options model))
        feature-ds (ds/select-columns dataset (:feature-columns model))
        label-columns (:target-columns model)
        thawed-model (thaw-model model model-def)
        pred-ds (predict-fn feature-ds
                            thawed-model
                            model)
        ]

    (if (= :classification (:model-type (meta pred-ds)))
      (-> (ds-mod/probability-distributions->label-column
           pred-ds (first label-columns))
          (ds/update-column (first label-columns)
                            #(vary-meta % assoc :column-type :prediction)))
      pred-ds)))


(defn explain
  "Explain (if possible) an ml model.  A model explanation is a model-specific map
  of data that usually indicates some level of mapping between features and importance"
  [model & [options]]
  (let [{:keys [explain-fn] :as model-def}
        (options->model-def (:options model))]
    (when explain-fn
      (explain-fn (thaw-model model model-def) model options))))


(defn default-loss-fn
  "Given a datset which must have exactly 1 inference target column return a default
  loss fn. If column is categorical, loss is tech.v3.ml.loss/classification-loss, else
  the loss is tech.v3.ml.loss/mae (mean average error)."
  [dataset]
  (let [target-ds (cf/target dataset)]
    (errors/when-not-errorf
     (== 1 (ds/column-count target-ds))
     "Dataset has more than 1 target specified: %d"
     (ds/column-count target-ds))
    (if (:categorical? (meta (first (vals target-ds))))
      loss/classification-loss
      loss/mae)))


(defn train-split
  "Train a model splitting the dataset using tech.v3.dataset.modelling/train-test-split
  and then calculate the loss using loss-fn.  Loss is added to the model map under :loss.

  * `loss-fn` defaults to loss/mae if target column is not categorical else defaults to
  loss/classification-loss."
  ([dataset options loss-fn]
   (let [{:keys [train-ds test-ds]} (ds-mod/train-test-split dataset options)
         target-colname (first (ds/column-names (cf/target train-ds)))
         model (train train-ds options)
         predictions (predict test-ds model)]
     (assoc model :loss (loss-fn (test-ds target-colname)
                                 (predictions target-colname)))))
  ([dataset options]
   (train-split dataset options (default-loss-fn dataset))))


(defn- do-k-fold
  [options loss-fn target-colname ds-seq]
  (let [models (mapv (fn [{:keys [train-ds test-ds]}]
                       (let [train-ds (preprocess train-ds options)
                             model (train train-ds options)
                             test-ds (preprocess test-ds options)
                             predictions (predict test-ds model)]
                         (assoc model :loss (loss-fn (predictions target-colname)
                                                     (test-ds target-colname)))))
                     ds-seq)
        loss-vec (mapv :loss models)
        {min-loss :min
         max-loss :max
         avg-loss :mean} (dfn/descriptive-statistics [:min :max :mean] loss-vec)
        min-model-idx (argops/argmin loss-vec)]
    (assoc (models min-model-idx)
           :min-loss min-loss
           :max-loss max-loss
           :avg-loss avg-loss
           :n-k-folds (count ds-seq))))


(defn train-k-fold
  "Train a model across k-fold datasets using tech.v3.dataset.modelling/k-fold-dataset
  and then calculate the min,max,and avg across results using loss-fn.  Adds
  :n-k-folds, :min-loss, :max-loss, :avg-loss and :loss (min-loss) to the
  model with the lowest loss.


  * `n-k-folds` defaults to 5.
  * `loss-fn` defaults to loss/mae if target column is not categorical else defaults to
     loss/classification-loss."

  ([dataset options n-k-folds loss-fn]
   (let [target-colname (first (ds/column-names (cf/target dataset)))]
     (->> (ds-mod/k-fold-datasets dataset n-k-folds options)
          (do-k-fold options loss-fn target-colname))))
  ([dataset options n-k-folds]
   (train-k-fold dataset options n-k-folds (default-loss-fn dataset)))
  ([dataset options]
   (train-k-fold dataset options 5 (default-loss-fn dataset))))


(defn train-auto-gridsearch
  "Train a model gridsearching across the options map.  The gridsearch map is built by
  merging the model's hyperparameter definitions into the options map.  If the sobol
  sequence returned has only one element a warning is issued.  Note this returns a
  sequence of models as opposed to a single model.


  * Searches across k-fold datasets if n-k-folds is > 1.  n-k-folds defaults to 5.
  * Searches (in parallel) through n-gridsearch option maps created via
    sobol-gridsearch.
  * Returns n-result-models (defaults to 5) sorted by avg-loss.
  * loss-fn can be provided or is the loss-fn returned via default-loss-fn."
  ([dataset options {:keys [n-k-folds
                            n-gridsearch
                            n-result-models
                            loss-fn]
                     :or {n-k-folds 5
                          n-gridsearch 75
                          n-result-models 5}
                     :as gridsearch-options}]

   (let [loss-fn (or loss-fn (default-loss-fn dataset))
         options (merge (hyperparameters (:model-type options)) options)
         gs-seq (take n-gridsearch (ml-gs/sobol-gridsearch options))
         target-colname (first (ds/column-names (cf/target dataset)))
         _ (when (== 1 (count gs-seq))
             (log/warn "Did not find any gridsearch axis in options map"))
         ds-seq (ds-mod/k-fold-datasets dataset n-k-folds gridsearch-options)]
     (->> gs-seq
          (pmap #(do-k-fold % loss-fn target-colname ds-seq))
          (sort-by :avg-loss)
          (take n-result-models)
          )))
  ([dataset options]
   (train-auto-gridsearch dataset options nil)))

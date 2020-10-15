(ns tech.v3.ml
  "Simple machine learning based on tech.v3.dataset functionality."
  (:require [tech.v3.datatype.errors :as errors]
            [tech.v3.dataset :as ds]
            [tech.v3.dataset.column-filters :as cf]
            [tech.v3.dataset.modelling :as ds-mod])
  (:import [java.util UUID]))


(defonce ^{:doc "Map of model kwd to model definition"} model-definitions* (atom nil))


(defn define-model!
  "Create a model definition.  An ml model is a function that takes a dataset and an options map and
  returns a model.  A model is something that, combined with a dataset, produces a inferred
  dataset."
  [model-kwd train-fn predict-fn {:keys [hyperparameter-map
                                         thaw-fn
                                         explain-fn]}]
  (swap! model-definitions* assoc model-kwd {:train-fn train-fn
                                             :predict-fn predict-fn
                                             :hyperparameters hyperparameter-map
                                             :thaw-fn thaw-fn
                                             :explain-fn explain-fn})
  :ok)

(defn model-definition-names
  []
  (keys @model-definitions*))


(defn options->model-def
  "Model definitions are specified by the :model-type keyword in the options map."
  [options]
  (if-let [model-def (get @model-definitions* (:model-type options))]
    model-def
    (errors/throwf "Failed to find model %s.  Is a require missing?" (:model-type options))))


(defn hyperparameters
  "Get the hyperparameters from the model-type in the options."
  [model-kwd]
  (:hyperparameters (options->model-def {:model-type model-kwd})))


(defn train
  "Given a dataset and an options map produce a model.  The model-type keyword in the options
  map selects which model definition to use to train the model.
  Returns a map containing at least:

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
  "Thaw a model.  Model's stored in options map may be 'frozen' meaning a 'thaw' operations is needed
  in order to use the model.  This happens for you during preduct but you may also cached the 'thawed'
  model on the model map under the ':thawed-model' keyword."
  [model {:keys [thaw-fn]}]
  (if-let [cached-model (get model :thawed-model)]
    cached-model
    (if thaw-fn
      (thaw-fn (get model :model-data))
      (get model :model-data))))


(defn predict
  "Predict returns a dataset with only the predictions in it.

  * For regression, a single column dataset is returned with the column named after the target
  * For classification, a dataset is returned with a float64 column for each target value and values
    that describe the probability distribution."
  [dataset model]
  (let [{:keys [predict-fn] :as model-def} (options->model-def (:options model))
        feature-ds (ds/select-columns dataset (:feature-columns model))
        label-columns (:target-columns model)
        thawed-model (thaw-model model model-def)
        pred-ds (-> (predict-fn feature-ds
                                thawed-model
                                model)
                    (ds/update-columnwise :all
                                          vary-meta assoc :column-type :prediction))]
    (if (= :classification (:model-type (meta pred-ds)))
      (ds-mod/probability-distributions->label-column pred-ds (first label-columns))
      pred-ds)))


(defn explain
  "Explain (if possible) an ml model.  A model explanation is a model-specific map
  of data that usually indicates some level of mapping between features and importance"
  [model & [options]]
  (let [{:keys [explain-fn] :as model-def}
        (options->model-def (:options model))]
    (when explain-fn
      (explain-fn (thaw-model model model-def) model options))))
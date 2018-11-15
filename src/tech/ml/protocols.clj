(ns tech.ml.protocols)


(defprotocol PMLSystem
  (system-name [system])
  (coalesce-options [system options]
    "Force double or float arrays by setting coalesce options.
See dataset/coalesce-dataset,
dataset/apply-dataset-options")
  (gridsearch-options [system options]
    "Given an options map which must include at least model-type, return a new
options map in the format that gridsearch requires.  Note these are general options,
documentation specific to each system will include much more extensive possible
options.")
  (train [system options coalesced-dataset]
    "Given these options return a model.  Model ideally is either a primitive byte array
or a clojure hash map.")
  (predict [system options model coalesced-dataset]
    "Predict the result given this model.  Regression predictions can be sequences of doubles
as can binary classification predictions.  Multiclass predictions should have each return a map
of class probabilities."))

(ns tech.ml.protocols)


(defprotocol PMLSystem
  (system-name [system])
  (coalesce-options [system]
    "Force double or float arrays by setting coalesce options.
See dataset/coalesce-dataset")
  (train [system options label-keys coalesced-dataset]
    "Given these options return a model.  Model ideally is either a primitive byte array
or a clojure hash map.")
  (predict [system options label-keys model coalesced-dataset]
    "Predict the result given this model.  Regression predictions can be sequences of doubles
as can binary classification predictions.  Multiclass predictions should have each return a map
of class probabilities."))

(ns tech.ml.protocols.system)


(defprotocol PMLSystem
  (system-name [system])
  (gridsearch-options [system options]
    "Given an options map which must include at least model-type, return a new
options map in the format that gridsearch requires.  Note these are general options,
documentation specific to each system will include much more extensive possible
options.")
  (train [system options dataset]
    "Given these options return a model.  Model ideally is either a primitive byte array
or a clojure hash map.")
  (thaw-model [system model]
    "Thaw a model from the return value of train")
  (predict [system options thawed-model dataset]
    "Predict the result given this model.  Results are depedent upon prediction type.
For regression, return a reader of predictions.  For classification, return a reader of
of posteriori probabilities"))


(defprotocol PMLExplain
  (explain-model [system model options]))


(defprotocol PInternalMLModelExplain
  (model-explain-model [model options]))


(extend-type Object
  PMLExplain
  (explain-model [this model options]
    (model-explain-model (thaw-model this model) options))
  PInternalMLModelExplain
  (model-explain-model [model options]
    (throw (Exception. (format "explain-model is unimplemented for %s"
                               (:model-type options))))))

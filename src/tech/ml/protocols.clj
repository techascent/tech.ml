(ns tech.ml.protocols)


(defprotocol PMLSystem
  (hyperparameters [system]
    "Get a map of option name to description.  Description includes
{:description text description of the hyperparameter
 :type (integer, float, enumeration)
 :range (for numeric)
 :classes (for enumerations)
 :default (default if not provided).
}
Used for auto-gridsearch.")

  (train [system feature-keys label-keys options-and-hyperparameters dataset]
    "dataset->model transformation.  Models should be something that converts
to/from nippy cleanly as well as at least somewhat viewable in repl.  Try to
avoid custom/opaque datatypes instead serialize them to byte arrays."))


(defprotocol PMLModel
  (predict [model dataset]
    "model->sequence of maps transformation"))

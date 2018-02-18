(ns gec.train
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [cheshire.core :refer :all]
            [aid.core :as aid]
            [clj-time.core :as t]
            [clj-time.format :as f]
            [com.rpl.specter :as s]
            [gec.helpers :as helpers]
            [gec.command :as command]))

(def current
  (f/unparse (f/formatter "yyyyMMddhhmmss") (t/now)))

(def to
  (helpers/join-paths "resources" "runs" current "hyperparameter.json"))

(defn get-from
  [python-boolean]
  (->> (str (if (read-string (str/lower-case python-boolean))
              "gpu"
              "cpu") ".json")
       (helpers/join-paths "resources" "hyperparameter")))

(defn add-commit
  [raw-json commit]
  (->> raw-json
       helpers/parse-keywordize
       (s/setval :commit commit)
       generate-string))

(defn copy-hyperparameter
  [python-boolean commit]
  (->> commit
       (add-commit (slurp (get-from python-boolean)))
       (helpers/spit-parents to)))

(def train*
  (partial helpers/python "gec/train.py" "--timestamp"))

(defn train
  [& more]
  (if (empty? more)
    (aid/mlet [python-boolean (helpers/python "gec/cuda.py")
               commit (command/git "rev-parse" "HEAD")]
              ;TODO randomly generate hyperparameters
              (copy-hyperparameter python-boolean commit)
              (train* current))
    (train* (first more))))
